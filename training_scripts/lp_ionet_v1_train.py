# %%
"""
 ## Imports
"""

# %%
import wandb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from transformers import AutoImageProcessor, MobileNetV1Model
from torch import nn
import torch
print("Importing lib....", flush=True)


SEED = 0

# set seed for all possible random functions to ensure reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# %%

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda:0"
print(f"Using {DEVICE} DEVICE", flush=True)

print("Libraries imported", flush=True)

# %%
print("Logging in to wandb\n\n", flush=True)
wandb.login()
wandb.init(project="smai-proj-lp_shadow-rem", config={
    "dataset": "Shadoc-768,1024",
    "architecture": "V2_LPNet+UNetWithAT",

    "n_images": 2000,
    "epochs": 200,
    "learning_rate": 0.0005,
    "batch_size": 16,
})
n_test = 400
config = wandb.config

# %%
""" Normalization class """

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
# unromalize the image


def unormalize_np_image(image, mean=norm_mean, std=norm_std):
    return np.clip((image * std + mean) * 255, 0, 255).astype(np.uint8)


# %%
""" ## Load Data """
print("Loading Data", flush=True)


def select_random_images(folder_path, k):
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.endswith(
        ('.jpg', '.jpeg', '.png', '.gif'))]

    if len(image_files) < k:
        return None

    return np.random.choice(range(len(image_files)), k, replace=False)


def load_images_from_folder(folder_path, selected_images_idx):
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.endswith(
        ('.jpg', '.jpeg', '.png', '.gif'))]
    images = []
    for i in selected_images_idx:
        filename = image_files[i]
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            images.append(Image.open(image_path))
    return images


folder_path_shadow_train = f"./train/input"
folder_path_removed_train = f"./train/target"
folder_path_shadow_test = f"./test/input"
folder_path_removed_test = f"./test/target"

n_train = config.n_images
img_train_indices = select_random_images(folder_path_shadow_train, n_train)
img_test_indices = select_random_images(folder_path_shadow_test, n_test)

img_shadow_raw_train = load_images_from_folder(
    folder_path_shadow_train, img_train_indices)
img_removed_raw_train = load_images_from_folder(
    folder_path_removed_train, img_train_indices)
img_shadow_raw_test = load_images_from_folder(
    folder_path_shadow_test, img_test_indices)
img_removed_raw_test = load_images_from_folder(
    folder_path_removed_test, img_test_indices)

del img_train_indices
del img_test_indices

transform = transforms.Compose([
    transforms.Resize((1024, 768)),  # Ensure the size
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=norm_mean, std=norm_std)
])

img_shadow_ = [transform(img) for img in img_shadow_raw_train]
del img_shadow_raw_train
X_train = torch.stack(img_shadow_)
del img_shadow_
print('Normalized train=>input images', flush=True)

img_removed_ = [transform(img) for img in img_removed_raw_train]
del img_removed_raw_train
Y_train = torch.stack(img_removed_)
del img_removed_
print('Normalized train=>target images', flush=True)

img_shadow_ = [transform(img) for img in img_shadow_raw_test]
del img_shadow_raw_test
X_test = torch.stack(img_shadow_)
del img_shadow_
print('Normalized test=>input images', flush=True)

img_removed_ = [transform(img) for img in img_removed_raw_test]
del img_removed_raw_test
Y_test = torch.stack(img_removed_)
del img_removed_
print('Normalized test=>target images', flush=True)

# X_train, X_test, Y_train, Y_test = train_test_split(img_shadow, img_removed, test_size=0.1)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, flush=True)
print("data loaded", flush=True)

# %%
"""
## Model Definition
"""

# %%
""" ## Layers Definition """


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

        self.skip_cnn = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        """
        Forward pass through the decoder block.

        Args:
                x (torch.Tensor): Input tensor.

        Returns:
                torch.Tensor: Output tensor.
        """
        temp_x = x
        x = self.cnn1(x)
        x = self.bnn1(x)
        x = self.relu(x)

        x = self.cnn2(x)
        x = self.bnn2(x)
        x = self.relu(x)

        x = self.cnn3(x)
        x = self.bnn3(x)

        # adding skip connection
        temp_x = self.skip_cnn(temp_x)

        x = x + temp_x

        if self.do_up_sampling:
            x = self.upsample(x)

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


class UnetWithAT(nn.Module):

    def __init__(self, lr=0.5):
        super(UnetWithAT, self).__init__()
        encoder_blocks, image_stem_layer, image_processor = get_encoder_layers()
        decoder_blocks = get_decoder_layers()

        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks

        self.image_processor = image_processor
        self.image_stem_layer = image_stem_layer
        self.lra = CoordAtt(3, 3)
        self.ldra = CoordAtt(3, 3)

        self.out_image_stem_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2,
                               bias=False, padding=1, output_padding=1),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.9997,
                           affine=True, track_running_stats=True),
            nn.ReLU()
        )

        # self.loss2 = nn.

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
            # print(f"Encoder block {indx} output shape: {x.shape}")
            enc_outputs.append(x)
        for indx, dec_block in enumerate(self.decoder_blocks):
            if indx == 0:
                x = dec_block(x)
            else:
                x = dec_block(
                    torch.cat([x, enc_outputs[len(self.decoder_blocks) - indx - 1]], dim=1))
            # print(f"Decoder block {indx} output shape: {x.shape}")

        # lra attention on skip connection
        temp_in_x = self.lra(temp_in_x)
        # ldra attention on output
        x = self.ldra(self.out_image_stem_layer(x))

        return x + temp_in_x

    def predict(self, x):
        with torch.no_grad():
            return self(x)


# %%
class LPNet(nn.Module):
    def __init__(self, i2it_model_path, L=2, interp_mode='nearest-exact', lr=0.1):
        super(LPNet, self).__init__()

        self.L = L
        self.interp_mode = interp_mode
        self.final_input_dim = None

        self.i2it_model = nn.DataParallel(
            UnetWithAT(), device_ids=[0, 1, 2, 3])
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

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, I0):
        # print(f'a forward loop')
        '''
            Downsampling the original image
        '''
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
        # print(f'Downsampled {IL.shape}')
        '''
            I2IT
        '''
        IL_cap = self.i2it_model.module.predict(IL)
        # print(f'I2IT: {IL_cap.shape}')
        '''
            Upsampling the translated image
        '''

        IL_up = nn.functional.interpolate(
            IL, size=[shapes[self.L-1][2], shapes[self.L-1][3]], mode=self.interp_mode)
        IL_cap_up = nn.functional.interpolate(
            IL_cap, size=[shapes[self.L-1][2], shapes[self.L-1][3]], mode=self.interp_mode)
        ResNet_inp = torch.cat([IL_up, IL_cap_up, H[self.L-1]], dim=1)

        # Ml_next = torch.mean(torch.stack([IL_up, IL_cap_up, H[self.L-1]]),dim=0,keepdim=True)[0]
        Ml_next = self.resid_refinement_net(ResNet_inp)

        H_ref = H[self.L-1]*Ml_next
        Il_cap = H_ref + IL_cap_up
#         print(H_ref.shape,H[self.L-1].shape,Ml_next.shape)

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
        with torch.no_grad():

            I0_cap_pred = self(I0.to(DEVICE))
            loss = self.loss_layer(I0_cap_pred, I0_cap.to(DEVICE))

            return loss.item()

    def predict(self, I0):
        with torch.no_grad():
            return self(I0.to(DEVICE))


# %%
"""
## Training  Functions
"""

# %%


def training(model, train_loader, n_epochs=3, validation=False, val_loader=None, print_every_epoch=True, optimizer=None):
    if print_every_epoch:
        print("Training started")

    for epoch in (range(n_epochs)):
        model.train()
        if print_every_epoch:
            print(f"Epoch {epoch+1}/{n_epochs}")
        running_loss = 0.0

        for x_train, y_train in tqdm(train_loader):
            y_pred = model(x_train)

            optimizer.zero_grad()
            loss = model.module.loss_layer(y_pred, y_train.to(DEVICE))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # epoch_loss += x_train.shape[0]*loss.item()

        metrices = {"train/loss": running_loss/len(train_loader),
                    "epoch": epoch+1,
                    }

        if validation:
            model.eval()
            metrices['val/loss'] = 0
            metrices['val/accuracy'] = 0
            for x_val, y_val in val_loader:
                metrices['val/loss'] += get_loss(model.module, x_val, y_val)
                metrices['val/accuracy'] += accuracy(
                    model.module, x_val, y_val)

            metrices['val/loss'] /= len(val_loader)
            metrices['val/accuracy'] /= len(val_loader)

            if epoch % 10 == 0 or epoch == n_epochs-1:

                x_vis = x_val
                y_vis = y_val
                # Extract the original image and convert it to numpy array
                original_image = x_vis[0].permute(1, 2, 0).cpu().numpy()
                gt_image = y_vis[0].permute(1, 2, 0).cpu().numpy()

                # Compute the predicted image using the model
                predicted_image = model.module.predict(x_vis[0].unsqueeze(
                    0)).cpu().detach().squeeze(0).permute(1, 2, 0).numpy()

                # unormalize the image
                original_image = unormalize_np_image(original_image)
                gt_image = unormalize_np_image(gt_image)
                predicted_image = unormalize_np_image(predicted_image)

                # Log the original image to Weights & Biases
                wandb.log({"original_image": [wandb.Image(
                    original_image, caption="Original Image")]})
                wandb.log(
                    {"gt_image": [wandb.Image(gt_image, caption="GT Image")]})
                wandb.log({"predicted_image": [wandb.Image(
                    predicted_image, caption="Predicted Image")]})

        wandb.log(metrices)

    if print_every_epoch:
        print("Training finished")
    wandb.finish()


def get_loss(model, x, y):
    return model.loss(x, y)


def accuracy(model, x, y):
    metric = SSIM().cuda()
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    with torch.no_grad():
        pred = model(x)
        return metric(pred, y)


# %%
"""
## Loading Model and Training
"""

# %%
torch.cuda.empty_cache()
model_lp = LPNet('./v2-shadrem-att.pth').to(DEVICE)
print("Model loaded", flush=True)

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
del train_dataset
print(f'initialized train dataloader', flush=True)


val_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=10, shuffle=False, drop_last=True)
del val_dataset
print(f'initialized validation dataloader', flush=True)

model = nn.DataParallel(model_lp, device_ids=[0, 1, 2, 3])
del model_lp


# %%
print("Training ...........", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
# training(model,train_loader, n_epochs=config.epochs, print_every_epoch=True, optimizer=optimizer)
training(model, train_loader, n_epochs=config.epochs, print_every_epoch=True,
         optimizer=optimizer, validation=True, val_loader=val_loader)

print("Training done", flush=True)

# %%
print("Saving model", flush=True)

# save the model
model_path = f"./v2_lpnet_shadrem-att.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully at path : {model_path}", flush=True)
torch.cuda.empty_cache()

# %%
