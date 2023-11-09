import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Generator(nn.Module):
    """Generator network with skip connection."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        # Initial convolution block
        self.init_conv = nn.Sequential(
            nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # Down-sampling layers
        self.down1 = Down(conv_dim  , conv_dim*2)
        self.down2 = Down(conv_dim*2, conv_dim*4)
        self.down3 = Down(conv_dim*4, conv_dim*8)

        # Bottleneck layers
        self.bottlenecks = nn.Sequential(
            *[ResidualBlock(dim_in=conv_dim*8, dim_out=conv_dim*8) for _ in range(repeat_num)]
        )

        # Up-sampling layers
        self.up1 = Up(conv_dim*8  , conv_dim*4)
        self.up2 = Up(conv_dim*4*2, conv_dim*2)
        self.up3 = Up(conv_dim*2*2, conv_dim  )
        
        # Output layer
        self.out = nn.Sequential(
            nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        # Initial convolution
        x = self.init_conv(x)

        # Down-sampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        # Bottleneck
        bottleneck = self.bottlenecks(d3)

        # Up-sampling
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d2], dim=1)) # NOTE remove this, too much preserved information
        up3 = self.up3(torch.cat([up2, d1], dim=1))

        # Output
        final_up = self.out(up3)
        return final_up

# class Generator(nn.Module):
#     """Generator network."""
#     def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
#         super(Generator, self).__init__()

#         layers = []
#         layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
#         layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
#         layers.append(nn.ReLU(inplace=True))

#         # Down-sampling layers.
#         curr_dim = conv_dim
#         for i in range(2):
#             layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
#             layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim * 2

#         # Bottleneck layers.
#         for i in range(repeat_num):
#             layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

#         # Up-sampling layers.
#         for i in range(2):
#             layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
#             layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim // 2

#         layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
#         layers.append(nn.Tanh())
#         self.main = nn.Sequential(*layers)

#     def forward(self, x, c):
#         # Replicate spatially and concatenate domain information.
#         # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
#         # This is because instance normalization ignores the shifting (or bias) effect.
#         c = c.view(c.size(0), c.size(1), 1, 1)
#         c = c.repeat(1, 1, x.size(2), x.size(3))
#         x = torch.cat([x, c], dim=1)
#         return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
