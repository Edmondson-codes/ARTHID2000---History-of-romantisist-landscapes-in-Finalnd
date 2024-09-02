import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Network Summary: https://medium.com/analytics-vidhya/what-is-unet-157314c87634

Encoder Blocks:
- each block consists of two 3 x 3 convolutions.
- each conv is are followed by a ReLU.
- The output of the ReLU acts as a skip connecttion for the corresponding decoder block

- Next 2x2 max pooling halves the dims of the feature map.

Bridge:
- Two 3x3 convs, where each is followed by a ReLU

Decoder Blocks:
- used to take abstract representation and generate a semantic segmentation mask.
- Starts with 2x2 transpose convolution
- ^ is concatinated with skip connection feature map from the corresponding encoder block'
- two 3x3 convolutions are used. Each is followed by ReLUs.

Useful:
- ConvTranspose2d
- MaxPool2d

'''

class ConvBlock(nn.Module):
    def __init__(self, in_pixels, out_pixels, middle_pixels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_pixels, middle_pixels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(middle_pixels, out_pixels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv_block(x)

class Encode(nn.Module):
    def __init__(self, in_pixels, out_pixels):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2),
        self.conv_block = ConvBlock(in_pixels, out_pixels, out_pixels)


    pass

class Decode(nn.Module):
    def __init__(self, in_pixels, out_pixels):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(in_pixels, in_pixels // 2, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_pixels, out_pixels, out_pixels)


    def forward(self, x, skip):
        x = self.conv_transpose(x)

        x_merged = torch.cat([skip, x])

        return self.conv_block(x_merged)


