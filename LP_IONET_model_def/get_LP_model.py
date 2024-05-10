from PIL import Image
import os
import torch
from torch import nn
import torchvision.transforms as transforms

from IOnet.IOnetv2.get_modelv2 import *
from IOnet.IOnetv2.get_modelv2 import *


class LPIONet(nn.Module):
    """
    The LP-IONET model class.
    """

    def __init__(self, i2it_model_path, L=2, interp_mode='nearest-exact', lr=0.1, Unet_class=UnetV2WithAT, DEVICE='cuda'):
        """
        Initializes the LPNet model.

        Args:
            i2it_model_path (str): The path to the pre-trained I2IT model.
            L (int, optional): The number of downsampling levels. Defaults to 2.
            interp_mode (str, optional): The interpolation mode used for upsampling. Defaults to 'nearest-exact'.
            lr (float, optional): The learning rate for optimization. Defaults to 0.1.
            Unet_class (nn.Module, optional): The class of the U-Net model used for I2IT. Defaults to UnetV2WithAT.
            DEVICE (str, optional): The device to be used for computation. Defaults to 'cuda'.

        Returns:
            None
        """
        super(LPIONet, self).__init__()
        self.DEVICE = DEVICE
        self.L = L
        self.interp_mode = interp_mode
        self.final_input_dim = None

        if DEVICE == 'cuda':
            self.i2it_model = nn.DataParallel(
                Unet_class(), device_ids=[0, 1, 2, 3])
        else:
            self.i2it_model = nn.DataParallel(
                Unet_class())

        self.i2it_model.load_state_dict(torch.load(i2it_model_path))
        for param in self.i2it_model.module.parameters():
            param.requires_grad_(False)
        self.i2it_model.eval()

        self.resid_refinement_net = nn.Sequential(
            # Depth-sequential network
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=1, groups=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
        )

        self.next_res_net = nn.ModuleList()
        for l in range(L-1):
            layers = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
            )

            self.next_res_net.append(layers)

        self.loss_layer = nn.L1Loss()

    def forward(self, I0):
        """
        Performs forward pass through the LPNet model.

        Args:
            I0 (Tensor): The input image tensor.

        Returns:
            Tensor: The output image tensor.
        """
        # Downsampling the original image
        H = []
        shapes = []
        IL = I0
        for l in range(self.L):
            curr_dim = IL.shape
            shapes.append(curr_dim)

            # print(f'Downsampling {IL.shape}')
            IL_next = nn.functional.interpolate(
                IL, size=[curr_dim[2]//2, curr_dim[3]//2], mode=self.interp_mode)
            IL_next_up = nn.functional.interpolate(
                IL_next, size=[curr_dim[2], curr_dim[3]], mode=self.interp_mode)
            Hl = IL - IL_next_up
            H.append(Hl)

            IL = IL_next

        self.final_input_dim = IL[0].shape

        # I2IT
        IL_cap = self.i2it_model.module.predict(IL)

        # Upsampling the translated image
        IL_up = nn.functional.interpolate(
            IL, size=[shapes[self.L-1][2], shapes[self.L-1][3]], mode=self.interp_mode)
        IL_cap_up = nn.functional.interpolate(
            IL_cap, size=[shapes[self.L-1][2], shapes[self.L-1][3]], mode=self.interp_mode)
        ResNet_inp = torch.cat([IL_up, IL_cap_up, H[self.L-1]], dim=1)

        # Ml_next = torch.mean(torch.stack([IL_up, IL_cap_up, H[self.L-1]]),dim=0,keepdim=True)[0]
        Ml_next = self.resid_refinement_net(ResNet_inp)

        H_ref = H[self.L-1]*Ml_next
        Il_cap = H_ref + IL_cap_up
        # print(H_ref.shape,H[self.L-1].shape,Ml_next.shape)

        for l in range(self.L-2, -1, -1):
            Ml_next_up = nn.functional.interpolate(
                Ml_next, size=[shapes[l][2], shapes[l][3]], mode=self.interp_mode)
            Ml_next = self.next_res_net[l](Ml_next_up)

            Il_cap_up = nn.functional.interpolate(
                Il_cap, size=[shapes[l][2], shapes[l][3]], mode=self.interp_mode)

            H_ref = H[l]*Ml_next
            Il_cap = H_ref + Il_cap_up

        return Il_cap

    def loss(self, I0, I0_cap,):
        """
        Calculates the loss between the predicted and ground truth images.

        Args:
            I0 (Tensor): The input image tensor.
            I0_cap (Tensor): The ground truth image tensor.

        Returns:
            float: The calculated loss value.
        """
        with torch.no_grad():

            I0_cap_pred = self(I0.to(self.DEVICE))
            loss = self.loss_layer(I0_cap_pred, I0_cap.to(self.DEVICE))

            return loss.item()

    def predict(self, I0):
        with torch.no_grad():
            return self(I0.to(self.DEVICE))
