import torch
import torch.nn as nn
from get_layers import get_encoder_layers, get_decoder_layers


class UnetWithoutAT(nn.Module):
    """
    U-Net model architecture.
    """

    def __init__(self):
        """
        Initializes the U-Net model.
        """
        super(UnetWithoutAT, self).__init__()
        encoder_blocks, image_stem_layer, image_processor = get_encoder_layers()
        decoder_blocks = get_decoder_layers()

        self.en1, self.en2, self.en3, self.en4, self.en5 = encoder_blocks
        self.de1, self.de2, self.de3, self.de4, self.de5 = decoder_blocks

        self.encoder_blocks = [self.en1, self.en2,
                               self.en3, self.en4, self.en5]
        self.decoder_blocks = [self.de1, self.de2,
                               self.de3, self.de4, self.de5]

        self.image_processor = image_processor
        self.image_stem_layer = image_stem_layer
        self.out_image_stem_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.9997,
                           affine=True, track_running_stats=True),
            nn.ReLU()
        )

    def forward(self, x, process_image=False):
        """
        Performs forward pass through the U-Net model.

        Args:
            x (torch.Tensor): Input image tensor.
            process_image (bool): Whether to preprocess input image.

        Returns:
            torch.Tensor: Output image tensor.
        """
        assert isinstance(x, torch.Tensor), "Input should be a tensor"
        if process_image:
            new_x = [self.image_processor(img)['pixel_values'][0] for img in x]
            x = torch.stack(new_x).permute(0, 3, 1, 2)
        assert x.shape[1] == 3, "Input image should have 3 channels(nx3x224x224)"

        x = self.image_stem_layer(x)
        enc_outputs = []
        for indx, enc_block in enumerate(self.encoder_blocks):
            x = enc_block(x)
            enc_outputs.append(x)
        for indx, dec_block in enumerate(self.decoder_blocks):
            if indx == 0:
                x = dec_block(x)
            else:
                x = dec_block(
                    torch.cat([x, enc_outputs[len(self.decoder_blocks) - indx - 1]], dim=1))

        return self.out_image_stem_layer(x)
