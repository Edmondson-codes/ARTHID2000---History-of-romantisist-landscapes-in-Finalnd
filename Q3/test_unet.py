import torch
import torch.nn
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
    def __init__(self, in_channels, out_channels):
        super.__init__()
        self.conv_block = nn.Sequential(
            nn.conv2d(in_channels, out_channels)
            nn.relu()
            nn.conv2d()
            nn.relu()
        )

class Encode(nn.Module):
    pass

class Decode(nn.Module):
    pass

