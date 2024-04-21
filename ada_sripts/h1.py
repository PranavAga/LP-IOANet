# %%
""" ## Imports
"""

import torch
from torch import nn
from transformers import AutoImageProcessor, MobileNetV1Model

from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
# import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb
import torch

SEED = 0

# set seed for all possible random functions to ensure reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True

DEVICE = "cuda:0"
print(f"Using {DEVICE} DEVICE",flush=True)


# %% 

wandb.login()
print("Training", flush=True)
wandb.init(project="smai-proj-unet", config={
    "dataset": "Shadoc-lowres",
    "architecture": "UNetWithoutAT",
    
    "epochs": 10,
    "learning_rate": 5e-1,
    "batch_size": 1,
})
config = wandb.config

# %%
""" ## Load Data """

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            images.append(img)
    return images

# folder_path_shadow = f"./input"
# folder_path_removed = f"./target"

# TODO for testing 
folder_path_shadow = f"../test_images/input"
folder_path_removed = f"../test_images/target"

n_images = 20
img_shadow = load_images_from_folder(folder_path_shadow)[:n_images]
img_removed = load_images_from_folder(folder_path_removed)[:n_images]

transform = transforms.Compose([
    transforms.Resize((256//2,192//2)),  # Ensure the size
    transforms.ToTensor(),          # Convert images to tensors
])

img_shadow = [transform(img) for img in img_shadow]
img_shadow = torch.stack(img_shadow)

img_removed = [transform(img) for img in img_removed]
img_removed = torch.stack(img_removed)

X_train, X_test, Y_train, Y_test = train_test_split(img_shadow, img_removed, test_size=0.1)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape,flush=True)

# %%
""" ## Layers Definition """

def get_encoder_layers():
    """
    Retrieves the layers of the MobileNetV1 model for encoding images.

    Returns:
        mobilenet_seq_blocks (list): List containing the layers of the MobileNetV1 model
            divided into blocks.
        conv_stem (torch.nn.Module): The stem convolutional layer of the MobileNetV1 model.
        image_processor (transformers.AutoImageProcessor): Pretrained image processor for
            MobileNetV1.
    """
    # download the model
    image_processor = AutoImageProcessor.from_pretrained(
        "google/mobilenet_v1_1.0_224")
    model = MobileNetV1Model.from_pretrained("google/mobilenet_v1_1.0_224")

    mobilenet_seq_blocks = nn.ModuleList()
    # block 1 will contain 4 layer of model.layer
    block = nn.Sequential(*list(model.layer)[:2])
    mobilenet_seq_blocks.append(block)

    block = nn.Sequential(*list(model.layer)[2:4])
    mobilenet_seq_blocks.append(block)

    block = nn.Sequential(*list(model.layer)[4:8])
    mobilenet_seq_blocks.append(block)

    block = nn.Sequential(*list(model.layer)[8:12])
    mobilenet_seq_blocks.append(block)

    block = nn.Sequential(*list(model.layer)[12:])
    mobilenet_seq_blocks.append(block)
    
    return mobilenet_seq_blocks, model.conv_stem, image_processor

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, do_up_sampling=True):
        # TODO: adding skip connections
        """
        Decoder block module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            expansion (int, optional): Expansion factor. Default is 3.
        """
        super(DecoderBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.cnn1 = nn.Conv2d(in_channels, in_channels *
                              expansion, kernel_size=1, stride=1)
        self.bnn1 = nn.BatchNorm2d(in_channels*expansion)

        # nearest neighbor x2
        self.do_up_sampling = do_up_sampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # DW conv/ c_in*exp x 5 x 5 x c_in*exp
        self.cnn2 = nn.Conv2d(in_channels*expansion, in_channels *
                              expansion, kernel_size=5, padding=2, stride=1)
        self.bnn2 = nn.BatchNorm2d(in_channels*expansion)

        self.cnn3 = nn.Conv2d(in_channels*expansion,
                              out_channels, kernel_size=1, stride=1)
        self.bnn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Forward pass through the decoder block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.cnn1(x)
        x = self.bnn1(x)
        x = self.relu(x)

        if self.do_up_sampling:
            x = self.upsample(x)

        x = self.cnn2(x)
        x = self.bnn2(x)
        x = self.relu(x)

        x = self.cnn3(x)
        x = self.bnn3(x)

        return x

def get_decoder_layers(out_sizes=[512, 256, 128, 64, 32]):
    decoder_blocks = nn.ModuleList()
    for i, out_size in enumerate(out_sizes):
        if i == 0:
            decoder_blocks.append(DecoderBlock(out_size*2, out_size))
        elif i == len(out_sizes)-1:
            decoder_blocks.append(DecoderBlock(
                out_size*4, out_size, do_up_sampling=False))
        else:
            decoder_blocks.append(DecoderBlock(out_size*4, out_size))
    return decoder_blocks

# %%
""" ## Model Definition """
loss_weights=(10,5)

class UnetWithoutAT(nn.Module):
    
    def __init__(self, loss_weights=(10,5), lr=0.5):
        super(UnetWithoutAT, self).__init__()
        encoder_blocks, image_stem_layer, image_processor = get_encoder_layers()
        decoder_blocks = get_decoder_layers()

        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks

        self.image_processor = image_processor
        self.image_stem_layer = image_stem_layer
        
        self.out_image_stem_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2,
                               bias=False, padding=1, output_padding=1),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.9997,
                           affine=True, track_running_stats=True),
            nn.ReLU()
        )

        # self.loss2 = nn.
        
        self.metric = SSIM()
        
    def forward(self, x, process_image=False):
        """
        Performs forward pass through the U-Net model.

        Args:
            x (torch.Tensor): Input image tensor.
            process_image (bool): Whether to preprocess input image.

        Returns:
            torch.Tensor: Output image tensor.
        """
        if process_image:
            new_x = [self.image_processor(img)['pixel_values'][0] for img in x]
            x = torch.stack(new_x).permute(0, 3, 1, 2)
        assert x.shape[1] == 3, "Input image should have 3 channels(nx3x224x224)"

        temp_in_x = x
        x = self.image_stem_layer(x)
        enc_outputs = []
        for indx, enc_block in enumerate(self.encoder_blocks):
            x = enc_block(x)
            print(f"Encoder block {indx} output shape: {x.shape}")
            enc_outputs.append(x)
        for indx, dec_block in enumerate(self.decoder_blocks):
            if indx == 0:
                x = dec_block(x)
            else:
                x = dec_block(
                    torch.cat([x, enc_outputs[len(self.decoder_blocks) - indx - 1]], dim=1))
            print(f"Decoder block {indx} output shape: {x.shape}")
        return 0.5 * self.out_image_stem_layer(x) + temp_in_x
    
# %%
""" ## Training Functions """

def training(model, train_loader, n_epochs=3, validation=False, val_loader=None, print_every_epoch=True):
    model = model.to(DEVICE)
    if print_every_epoch: print("Training started")
    
    for epoch in (range(n_epochs)):
        if print_every_epoch: print(f"Epoch {epoch+1}/{n_epochs}")
        
        for x_train, y_train in tqdm(train_loader):
            y_pred = model(x_train)
            loss = loss_layer(model,y_pred,y_train)
            loss.backward()
            model.optimizer.zero_grad()
            loss = loss_layer(model,y_pred,y_train)
            
            metrices = {"train/loss": loss.item(),
                        "epoch": epoch+1, 
            }
            if validation:
                metrices['val/loss'] = 0
                metrices['val/accuracy'] = 0
                for x_val, y_val in val_loader:
                    metrices['val/loss'] += get_loss(model,x_val,y_val)
                    metrices['val/accuracy'] += accuracy(model,x_val,y_val)
                
                metrices['val/loss'] /= len(val_loader)
                metrices['val/accuracy'] /= len(val_loader)
                
                if print_every_epoch: print(f"train_loss: {metrices['train/loss']}, val_loss: {metrices['val/loss']}, val_accuracy: {metrices['val/accuracy']}")

            wandb.log(metrices)                
            loss.backward()
            model.optimizer.step()
            
    if print_every_epoch: print("Training finished")
    
    wandb.finish()
    return model.loss(x_train,y_train)

def loss_layer(model, y_pred, y):
    loss1 = nn.L1Loss().cuda()
    y_pred = y_pred.to(DEVICE)
    y = y.to(DEVICE)
    '''
    Considering weighted loss
    '''
    return loss_weights[0] * loss1(y_pred, y)

def get_loss(model, x, y):
    with torch.no_grad():
        pred = model(x)
        return loss_layer(model, pred, y).item()

def accuracy(model, x, y):
    with torch.no_grad():
        pred = model(x)
        return model.metric(pred, y)
    
# %%
""" ## Loading Model and Training """
torch.cuda.empty_cache()
model = UnetWithoutAT()

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

model = nn.DataParallel(model, device_ids=[0,1,2,3])

# %%

training(model,train_loader, n_epochs=config.epochs, print_every_epoch=True)

# %%
