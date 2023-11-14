import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, channels, size) -> None:
        """
        Args:
            channels: Channel dimension.
            size: Current image resolution.
        """
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False) -> None:
        super(DoubleConv, self).__init__()
        self.residual = residual

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(1, mid_channels),
            nn.InstanceNorm2d(mid_channels, affine=True, track_running_stats=True),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(1, out_channels),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
        )
    def forward(self, x):
        if self.residual:
            # return F.gelu(x + self.double_conv(x))
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)

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
        # self.down = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
        #     nn.ReLU(inplace=True)
        # )
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), # each downblock will reduce the input size by 2
            # DoubleConv(in_channels, in_channels, residual=False),
            # DoubleConv(in_channels, out_channels),
            DoubleConv(in_channels, out_channels, residual=False),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # self.up = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
        #     nn.ReLU(inplace=True)
        # )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            # DoubleConv(in_channels, in_channels, residual=False),
            # DoubleConv(in_channels, out_channels, in_channels // 2),
            DoubleConv(in_channels, out_channels, residual=False),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1) # apply skip connection
        x = self.conv(x)
        return x

class Generator(nn.Module):
    """Generator network with skip connection."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        # # Initial convolution block
        self.init_conv = nn.Sequential(
            nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.01)
        )

        # # Down-sampling layers
        # self.down1 = Down(conv_dim  , conv_dim*2)
        # self.down2 = Down(conv_dim*2, conv_dim*4)
        # self.down3 = Down(conv_dim*4, conv_dim*8)

        # # Bottleneck layers
        # self.bottlenecks = nn.Sequential(
        #     *[ResidualBlock(dim_in=conv_dim*8, dim_out=conv_dim*8) for _ in range(repeat_num)]
        # )

        # # Up-sampling layers
        # self.up1 = Up(conv_dim*8  , conv_dim*4)
        # self.up2 = Up(conv_dim*4*2, conv_dim*2)
        # self.up3 = Up(conv_dim*2*2, conv_dim  )
        
        # Output layer
        self.out = nn.Sequential(
            nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

        # Encoder
        self.inc = DoubleConv(3 + c_dim, 64)
        self.down1 = Down(64, 128)          # 64 if input 128
        #self.sa1 = SelfAttention(128, 32)   
        self.down2 = Down(128, 256)         # 32
        #self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 512)         # 16
        #self.sa3 = SelfAttention(256, 8)

        # Bottleneck
        self.bot1 = DoubleConv(512, 512, residual=True)
        self.bot2 = DoubleConv(512, 512, residual=True)
        self.bot3 = DoubleConv(512, 512, residual=True)
        self.bot4 = DoubleConv(512, 512, residual=True)
        self.bot5 = DoubleConv(512, 512, residual=True)
        self.bot6 = DoubleConv(512, 256)

        # Decoder
        self.up1 = Up(512, 128)
        #self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        #self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        #self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        # # Initial convolution
        # x = self.init_conv(x)

        # # Down-sampling
        # d1 = self.down1(x)
        # d2 = self.down2(d1)
        # d3 = self.down3(d2)

        # # Bottleneck
        # bottleneck = self.bottlenecks(d3)

        # # Up-sampling
        # up1 = self.up1(bottleneck)
        # up2 = self.up2(torch.cat([up1, d2], dim=1)) # NOTE remove this, too much preserved information
        # up3 = self.up3(torch.cat([up2, d1], dim=1))

        # # Output
        # final_up = self.out(up3)
        # return final_up
        x1 = self.init_conv(x)
        x2 = self.down1(x1)
        #x2 = self.sa1(x2)
        x3 = self.down2(x2)
        #x3 = self.sa2(x3)
        x4 = self.down3(x3)
        #x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        x4 = self.bot4(x4)
        x4 = self.bot5(x4)
        x4 = self.bot6(x4)

        x = self.up1(x4, x3)
        #x = self.sa4(x)
        x = self.up2(x, x2)
        #x = self.sa5(x)
        x = self.up3(x, x1)
        
        #x = self.sa6(x)
        output = self.out(x)
        return output

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
