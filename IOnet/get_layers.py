import torch
from transformers import AutoImageProcessor, MobileNetV1Model
import torch.nn as nn


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
        print("Alert: skip connection is not implemented in the decoder block")
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
