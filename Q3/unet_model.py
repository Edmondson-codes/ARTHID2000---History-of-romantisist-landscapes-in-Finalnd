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
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_pixels, out_pixels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class Encode(nn.Module):
    def __init__(self, in_pixels, out_pixels):
        super().__init__()

        self.pool_and_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(in_pixels, out_pixels, out_pixels)
        )

    def forward(self, x):
        return self.pool_and_conv(x)


class Decode(nn.Module):
    def __init__(self, in_pixels, out_pixels):
        super().__init__()

        self.upsamp = nn.Upsample(2, mode='bilinear', align_corners=False)
        self.conv_block = ConvBlock(in_pixels, out_pixels, out_pixels)

    def forward(self, x, skip):
        print(f"x before upsampling: {x.size()}")
        x = self.upsamp(x)
        print(f"x after upsampling: {x.size()}")

        # Calculate padding needed
        diff_x = skip.size()[2] - x.size()[2]
        diff_y = skip.size()[3] - x.size()[3]
        print(f"Padding to be applied: {[diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]}")

        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        print(f"x after padding: {x.size()}")

        x_merged = torch.cat([skip, x], dim=1)
        print(f"x_merged after concatenation: {x_merged.size()}")

        return self.conv_block(x_merged)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.out_conv(x)


class UNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.first = ConvBlock(num_channels, 64, 64)
        self.encode1 = Encode(64, 128)
        self.encode2 = Encode(128, 256)
        self.encode3 = Encode(256, 512)
        self.bridge = Encode(512, 512)
        self.decode1 = Decode(512, 256)
        self.decode2 = Decode(256, 128)
        self.decode3 = Decode(128, 64)
        self.decode4 = Decode(64, 32)
        self.out = OutConv(32, num_classes)

    def forward(self, x):
        print('iteration')
        x1 = self.first(x)
        x2 = self.encode1(x1)
        x3 = self.encode2(x2)
        x4 = self.encode3(x3)
        x5 = self.bridge(x4)

        x = self.decode1(x5, x4)
        x = self.decode2(x, x3)
        x = self.decode3(x, x2)
        x = self.decode4(x, x1)
        out = self.out(x)
        print('iteration end')
        return out
