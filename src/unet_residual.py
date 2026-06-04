import torch
import torch.nn as nn
from unet import DoubleConv, DownSample, UpSample, UpSample_last

class UNetResidual(nn.Module):
    """UNet architecture for image super-resolution.

    When `residual=True` the network predicts a residual which is added to a
    bilinear upsample of the LR input. This often improves pixel-wise metrics
    (e.g. when using SmoothL1/SmoothLoss) because the model only needs to
    predict high-frequency corrections.
    """
    def __init__(self, in_channels, num_classes, align_corners: bool = True):
        super().__init__()
        self.align_corners = align_corners
        self.first_up_convolution_1 = nn.ConvTranspose2d(in_channels, 32, kernel_size=3, stride=3)

        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512,1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.last_up_convolution = UpSample_last(64, 32)

        self.out = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)

    def forward(self, x):

        first_1 = self.first_up_convolution_1(x)

        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)
        up_5 = self.last_up_convolution(up_4,first_1)

        out = self.out(up_5)
        interp = torch.nn.functional.interpolate(
            x,
            size=out.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners,
        )
        out = interp + out
        return out


