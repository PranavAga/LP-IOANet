import torch
from transformers import AutoImageProcessor, MobileNetV1Model
from transformers import MobileNetV2Config, MobileNetV2Model
import torch.nn as nn

# # Initializing a "mobilenet_v2_1.0_224" style configuration
# configuration = MobileNetV2Config()

# # Initializing a model from the "mobilenet_v2_1.0_224" style configuration
# model = MobileNetV2Model(configuration)
# model_layers = model.layer


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
