# %%
"""
## Imports
"""

# %%
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

# %%
SEED = 0

# set seed for all possible random functions to ensure reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True

# %%
wandb.login()

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} DEVICE",flush=True)

# %%
"""
## Model
"""

# %%
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

    mobilenet_seq_blocks = []
    # block 1 will contain 4 layer of model.layer
    block = nn.Sequential(*list(model.layer)[:2])
    mobilenet_seq_blocks.append(block)
    # print("-"*30,"\n\nblock 1:", block)

    block = nn.Sequential(*list(model.layer)[2:4])
    mobilenet_seq_blocks.append(block)
    # print("-"*30,"\n\nblock 2:", block)

    block = nn.Sequential(*list(model.layer)[4:8])
    mobilenet_seq_blocks.append(block)
    # print("-"*30,"\n\nblock 3:", block)

    block = nn.Sequential(*list(model.layer)[8:12])
    mobilenet_seq_blocks.append(block)
    # print("-"*30,"\n\nblock 4:", block)

    block = nn.Sequential(*list(model.layer)[12:])
    mobilenet_seq_blocks.append(block)
    # print("-"*30,"\n\nblock 5:", block)

    # printing the input and output channels of the first and last layers of each block
    # for i, block in enumerate(mobilenet_seq_blocks):
    # 	# Extracting the first and last layers of the block
    # 	first_layer = block[0]
    # 	last_layer = block[-1]

    # 	# Get the input and output channels of the first and last layers
    # 	input_channel_first_layer = first_layer.convolution.in_channels
    # 	output_channel_last_layer = last_layer.convolution.out_channels

    # 	print(f"Block {i + 1}:")
    # 	print("Input channel of the first layer:", input_channel_first_layer)
    # 	print("Output channel of the last layer:", output_channel_last_layer)

    return mobilenet_seq_blocks, model.conv_stem, image_processor


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=3, do_up_sampling=True):
        # print("Alert: skip connection is not implemented in the decoder block")
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
    decoder_blocks = []
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

class UnetWithoutAT(nn.Module):
    """
    U-Net model architecture.
    """

    def __init__(self, loss_weights=(10,5), lr=0.5):
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
        # print("Image stem layer:", self.image_stem_layer)
        self.out_image_stem_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2,
                               bias=False, padding=1, output_padding=1),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.9997,
                           affine=True, track_running_stats=True),
            nn.ReLU()
        )

        self.loss_weights = loss_weights
        self.loss1 = nn.L1Loss()
        # self.loss2 = nn.

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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

        temp_in_x = x
        x = self.image_stem_layer(x)
        enc_outputs = []
        for indx, enc_block in enumerate(self.encoder_blocks):
            x = enc_block(x)
            # print(f"Encoder block {indx} output shape: {x.shape}")
            enc_outputs.append(x)
        for indx, dec_block in enumerate(self.decoder_blocks):
            if indx == 0:
                x = dec_block(x)
            else:
                x = dec_block(
                    torch.cat([x, enc_outputs[len(self.decoder_blocks) - indx - 1]], dim=1))
            # print(f"Decoder block {indx} output shape: {x.shape}")
        return 0.5 * self.out_image_stem_layer(x) + temp_in_x
    
    def loss_layer(self,y_pred,y):
        '''
        Considering wieghted loss
        '''
        return self.loss_weights[0]*self.loss1(y_pred, y)
    
    def loss(self,x,y):
        with torch.no_grad():
            pred = self(x)
            return self.loss_layer(pred,y).item()
    
    def accuracy(self,x,y):
        with torch.no_grad():
            pred = self(x)
            metric = SSIM()
            if DEVICE=='cuda':
                metric = metric.cuda()
            return metric(pred,y)
    
    def fit(self, x_train, y_train, batch_size=32, n_epochs=3, validation=False, x_val=None, y_val=None):
        '''
        
        '''
        '''
        wandb.log(metrics)

        # store best model
        
    wandb.finish()
        '''
        for epoch in tqdm(range(n_epochs)):
            #mini-batch gradient descent
            for i in range((x_train.shape[0] - 1) // batch_size + 1):
                start_i = i * batch_size
                end_i = start_i + batch_size
                xb = x_train[start_i:end_i]
                yb = y_train[start_i:end_i]
                pred = self(xb)

                self.optimizer.zero_grad()
                loss = self.loss_layer(pred,yb)
               
                metrices = {"train/loss": loss.item(),
                           "epoch": epoch+1, 
                }
                if validation:
                    metrices['val/loss'] = self.loss(x_val,y_val)
                    metrices['val/accuracy'] = self.accuracy(x_val,y_val)
                    
                wandb.log(metrices)
                
                loss.backward()
                self.optimizer.step()
        
        wandb.finish()
        return self.loss(x_train,y_train)

# %%
"""
## Loading data
"""

# %%
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            images.append(img)
    return images

folder_path_shadow = f"./input"
folder_path_removed = f"./target"

n_images = 10
img_shadow = load_images_from_folder(folder_path_shadow)[:n_images]
img_removed = load_images_from_folder(folder_path_removed)[:n_images]

# %%
print(len(img_shadow),flush=True)
print(img_shadow[0].size,flush=True)

print(len(img_removed),flush=True)
print(img_removed[0].size,flush=True)

# %%
transform = transforms.Compose([
    transforms.Resize((256,192)),  # Ensure the size
    transforms.ToTensor(),          # Convert images to tensors
])

# %%
# Convert PIL images to tensors
img_shadow = [transform(img) for img in img_shadow]
img_shadow = torch.stack(img_shadow)

img_removed = [transform(img) for img in img_removed]
img_removed = torch.stack(img_removed)

# %%
X_train, X_test, Y_train, Y_test = train_test_split(img_shadow, img_removed, test_size=0.1)
print(X_train.shape,Y_test.shape,flush=True)

# %%
if DEVICE=='cuda':
    X_train = X_train.cuda()
    Y_train = Y_train.cuda()
    
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()
    

# %%
"""
## Training
"""

# %%
wandb.init(project="smai-proj-unet",config={
    "dataset": "Shadoc-lowres",
    "architecture": "UNetWithoutAT",
    
    "epochs":10,
    "learning_rate" : 5e-1,
    "batch_size": 16,
      })
config = wandb.config

# %%
model = UnetWithoutAT(lr=config.learning_rate)
if DEVICE=='cuda':
    model = model.cuda()

# %%
model.fit(X_train,Y_train,batch_size=config.batch_size, n_epochs=config.epochs, validation=True, x_val=X_test, y_val=Y_test)

# %%
