# %% """ ## Imports """
from torch.optim.lr_scheduler import StepLR
import lpips
import PIL
import wandb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, MobileNetV1Model
from transformers import MobileNetV2Config, MobileNetV2Model
from torch import nn
import torch
print("\n>\t", "Importing lib....", flush=True)

SEED = 0

# set seed for all possible random functions to ensure reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda:0"
print(f"Using {DEVICE} DEVICE", flush=True)

print("\n>\t", "Libraries imported", flush=True)
# %% """ ## Wandb Login and Initialize ""
print("\n>\t", "Logging in to wandb\n\n", flush=True)
wandb.login()
print("Training", flush=True)
wandb.init(project="smai-proj-shadow-rem", config={
    "dataset": "Shadoc-192,256",
    "architecture": "UNetWithAT",
    "log_images_interval": 50,
    "log_images_indices": [0, 7, 20, 30, 40],
    "test_ratio": 0.05,
    "epochs": 500,
    "learning_rate": 0.002,
    "step_size": 10,
    "gamma": 0.95,
    "batch_size": 32,
})
config = wandb.config

# %%  """ Normalization class """

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
# unromalize the image


def unormalize_np_image(image, mean=norm_mean, std=norm_std):
    return np.clip((image * std + mean) * 255, 0, 255).astype(np.uint8)


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            images.append(img)
    return images

# %% """ ## Load Data """
# print("\n>\t","Loading Data", flush=True)


# # folder_path_shadow = f"./input"
# # folder_path_removed = f"./target"

# folder_path_shadow = f"../data/192,256/train/input"
# folder_path_removed = f"../data/192,256/train/target"

# # n_images = 2000
# # img_shadow = load_images_from_folder(folder_path_shadow)[:n_images]
# # img_removed = load_images_from_folder(folder_path_removed)[:n_images]

# img_shadow_raw = load_images_from_folder(folder_path_shadow)
# img_removed_raw = load_images_from_folder(folder_path_removed)

# transform = transforms.Compose([
#     transforms.Resize((256,192)),  # Ensure the size
#     transforms.ToTensor(),          # Convert images to tensors
#     transforms.Normalize(mean=norm_mean, std=norm_std)
# ])

# img_shadow_ = [transform(img) for img in img_shadow_raw]
# img_shadow = torch.stack(img_shadow_)
# del img_shadow_
# del img_shadow_raw
# print('Normalized input images',flush=True)

# img_removed_ = [transform(img) for img in img_removed_raw]
# img_removed = torch.stack(img_removed_)
# del img_removed_
# del img_removed_raw
# print('Normalized target images',flush=True)


# X_train, X_test, Y_train, Y_test = train_test_split(img_shadow, img_removed, test_size=config.test_ratio, random_state=SEED)

# X_train = torch.tensor(X_train)
# X_test = torch.tensor(X_test)
# Y_train = torch.tensor(Y_train)
# Y_test = torch.tensor(Y_test)

# print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape,flush=True)
# print("\n>\t","data loaded", flush=True)

# %% """ ## Save the data """
# # dump the data
# torch.save(X_train, "./temp_data/X_train.pth")
# torch.save(X_test, "./temp_data/X_test.pth")
# torch.save(Y_train, "./temp_data/Y_train.pth")
# torch.save(Y_test, "./temp_data/Y_test.pth")

# %% """ ## loading temp files """
print("\n>\t", "loading temp files", flush=True)
X_train = torch.load("./temp_data/X_train.pth")
X_test = torch.load("./temp_data/X_test.pth")
Y_train = torch.load("./temp_data/Y_train.pth")
Y_test = torch.load("./temp_data/Y_test.pth")

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, flush=True)

# %% """ ## Layers Definition """
print("\n>\t", "loading layers", flush=True)


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

        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the decoder block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        temp_x = self.identity(x)
        x = self.cnn1(x)
        x = self.bnn1(x)
        x = self.relu(x)

        if self.do_up_sampling:
            x = self.upsample(x)
            temp_x = self.upsample(temp_x)

        x = self.cnn2(x)
        x = self.bnn2(x)
        x = self.relu(x)

        x = self.cnn3(x)
        x = self.bnn3(x)

        return x + temp_x


def get_encoderv2_layers():
    """
    Returns the encoder layers of the MobileNetV2 model.
    """
    configuration = MobileNetV2Config()
    model = MobileNetV2Model(configuration)

    list_en = nn.ModuleList()
    encoder_layers_idx = [
        [0],
        [1, 2],
        [3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
    ]
    for idx, layer in enumerate(encoder_layers_idx):
        list_en.append(nn.Sequential(*[model.layer[i] for i in layer]))
    list_en.append(model.conv_1x1)
    return list_en, model.conv_stem


def get_decoderv2_layers():
    """
    Returns the decoder layers of the MobileNetV2 model based on the encoder layers.
    """
    decoder_layers = nn.ModuleList()
    decoder_layers.append(DecoderBlock(1280, 320, do_up_sampling=False))
    decoder_layers.append(DecoderBlock(320+320, 160, do_up_sampling=False))
    decoder_layers.append(DecoderBlock(160+160, 96, do_up_sampling=True))
    decoder_layers.append(DecoderBlock(96+96, 64, do_up_sampling=False))
    decoder_layers.append(DecoderBlock(64+64, 32, do_up_sampling=True))
    decoder_layers.append(DecoderBlock(32+32, 24, do_up_sampling=True))
    decoder_layers.append(DecoderBlock(24+24, 16, do_up_sampling=True))

    # a conv layer to get the final output
    out_stem = nn.Sequential(
        nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(1, 1),
                               stride=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.997,
                           affine=True, track_running_stats=True),
            nn.ReLU6(),
        ),
        nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(
                1, 1), groups=32, bias=False, padding=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.997,
                           affine=True, track_running_stats=True),
            nn.ReLU6(),
        ),
        nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(
                2, 2), bias=False, padding=1, output_padding=1),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.997,
                           affine=True, track_running_stats=True),
        )
    )

    return decoder_layers, out_stem


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

# %% """ ## Model Definition """


class UnetV2WithAT(nn.Module):
    def __init__(self, lr=0.5):
        super(UnetV2WithAT, self).__init__()
        self.encoder_blocks, self.first_layer = get_encoderv2_layers()
        self.decoder_blocks, self.last_layer = get_decoderv2_layers()

        self.lra = CoordAtt(3, 3)
        self.ldra = CoordAtt(3, 3)

    def forward(self, x):
        """
        Performs forward pass through the U-Net model.

        Args:
                x (torch.Tensor): Input image tensor.
                process_image (bool): Whether to preprocess input image.

        Returns:
                torch.Tensor: Output image tensor.
        """
        assert x.shape[1] == 3, "input image should have 3 channels(nx3x224x224)"

        temp_in_x = x

        x = self.first_layer(x)

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

        x = self.last_layer(x)

        # lra attention on skip connection
        temp_in_x = self.lra(temp_in_x)
        # ldra attention on output
        x = self.ldra(x)

        return x + temp_in_x

    def predict(self, x):
        with torch.no_grad():
            return self(x)


# %% """ ## Loss Functions """
print("\n>\t", "loading loss functions", flush=True)
# loss functions
loss1 = nn.L1Loss().to(DEVICE)
lpips_layer = lpips.LPIPS(net='alex').to(DEVICE)
print("\n>\t", "loss functions loaded", flush=True)
# %% """ Training Functions """


def training(model, train_loader, n_epochs=3, validation=False, val_loader=None, print_every_epoch=True, optimizer=None):
    if print_every_epoch:
        print("Training started")

    # log the test gt and orginal image to wandb
    for idx in config.log_images_indices:
        # Extract the images at the specified index
        original_image = X_test[idx].permute(1, 2, 0).cpu().numpy()
        gt_image = Y_test[idx].permute(1, 2, 0).cpu().numpy()

        # Unnormalize the images
        original_image = unormalize_np_image(original_image)
        gt_image = unormalize_np_image(gt_image)

        # Log the images to Weights & Biases with clear captions
        wandb.log({"original_image_" + str(idx): [wandb.Image(original_image, caption="|Original index: " + str(idx) + "|")]})
        wandb.log({"gt_image_" + str(idx): [wandb.Image(gt_image, caption="|GT index: " + str(idx) + "|")]})

    for epoch in (range(n_epochs)):
        model.train()
        if print_every_epoch:
            print(f"Epoch {epoch+1}/{n_epochs}")
        running_loss = 0.0

        for x_train, y_train in tqdm(train_loader):
            y_pred = model(x_train)

            optimizer.zero_grad()
            loss = loss_layer(model, y_pred, y_train)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # epoch_loss += x_train.shape[0]*loss.item()

        metrices = {"train/loss": running_loss/len(train_loader),
                    "epoch": epoch+1,
                    }

        if validation:
            model.eval()
            with torch.no_grad():
                metrices['val/loss'] = 0
                metrices['val/accuracy'] = 0
                for x_val, y_val in val_loader:
                    metrices['val/loss'] += get_loss(model, x_val, y_val)
                    metrices['val/accuracy'] += accuracy(model, x_val, y_val)

                metrices['val/loss'] /= len(val_loader)
                metrices['val/accuracy'] /= len(val_loader)

                wnb_image_log_indices = config.log_images_indices

                if epoch % config.log_images_interval == 0:
                    for idx in wnb_image_log_indices:
                        # Compute the predicted image using the model
                        predicted_image = model.module.predict(X_test[idx].unsqueeze(
                            0)).cpu().detach().squeeze(0).permute(1, 2, 0).numpy()

                        # Unnormalize the images
                        predicted_image = unormalize_np_image(predicted_image)

                        # Log the images to Weights & Bia   ses with clear captions
                        wandb.log({"predicted_image_" + str(idx): [wandb.Image(
                            predicted_image, caption="|Predicted index: " + str(idx) + "|")]})

                # if print_every_epoch: print(f"Epoch {epoch}/{n_epochs}: |>train_loss: {metrices['train/loss']}, val_loss: {metrices['val/loss']}, val_accuracy: {metrices['val/accuracy']}")
                # # print image
                # if print_every_epoch:
                #     print("Original Image")
                #     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                #     ax[0].imshow(original_image)
                #     ax[1].imshow(gt_image)
                #     ax[2].imshow(predicted_image)
                #     for a in ax:
                #         a.axis('off')
                #     plt.show()

        # scheduler to decrease learning rate
        scheduler.step()
        wandb.log(metrices)

    if print_every_epoch:
        print("Training finished")
    wandb.finish()


def loss_layer(model, y_pred, y):
    y_pred = y_pred.to(DEVICE)
    y = y.to(DEVICE)
    '''
    Considering weighted loss
    '''
    # lpips loss mean as it return array of scores
    lpips_loss = lpips_layer.forward(y_pred, y)
    mean_lpips_loss = torch.mean(lpips_loss)

    return loss_weights[0] * loss1(y_pred, y) + loss_weights[1] * mean_lpips_loss


def get_loss(model, x, y):
    with torch.no_grad():
        pred = model(x)
        return loss_layer(model, pred, y).item()


def accuracy(model, x, y):
    metric = SSIM().to(DEVICE)
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    with torch.no_grad():
        pred = model(x)
        return metric(pred, y)


# %% """ ## Loading Model and Training """
print("\n>\t", "loading Models and making dataloaders", flush=True)

torch.cuda.empty_cache()
model = UnetV2WithAT().to(DEVICE)

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
val_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False)

model = nn.DataParallel(model, device_ids=[0, 1])

print("\n>\t", "Model loaded", flush=True)
# %% """ Training """
print("\n>\t", "Training ...........", flush=True)

loss_weights = (10, 5)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

# training(model,train_loader, n_epochs=config.epochs, print_every_epoch=True, optimizer=optimizer)
training(model, train_loader, n_epochs=config.epochs, print_every_epoch=False,
         optimizer=optimizer, validation=True, val_loader=val_loader)

print("\n>\t", "Training done", flush=True)
# %%
print("\n>\t", "Saving model", flush=True)

# save the model with epoch done in the name
""" Save the model """
# save the model
model_path = f"./shadrem-att.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully at path : {model_path}", flush=True)

# %%
torch.cuda.empty_cache()

# %%