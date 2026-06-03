import torch.nn as nn

# model
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x

class BN_PReLU(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.prelu(x)
        return x

class TransposeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        x = self.conv_transpose(x)
        return x
    
class Interpolate(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=True):
        super().__init__()
        self.interpolate = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    def forward(self, x):
        x = self.interpolate(x)
        return x

class LoGSRN(nn.Module):
    """LoGSRN architecture for image super-resolution."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.initial_conv = Conv(in_channels, 128, kernel_size=9, padding=4)
        self.interp = Interpolate(scale_factor=3, mode='bilinear', align_corners=True)
        self.rconv=Conv(128, 128, kernel_size=3, padding=1)
        self.bn_prelu = BN_PReLU(128)
        self.transpose_conv = TransposeConv(128, 128, kernel_size=3, stride=3, padding=0, output_padding=0)
        self.conv_128x64K3 = Conv(128, 64, kernel_size=3, padding=1)
        self.conv_64xinK9 = Conv(64, in_channels, kernel_size=9, padding=4)
        self.conv_inx128K3 = Conv(in_channels, 128, kernel_size=3, padding=1)

    def forward(self, x):
        interp = self.interp(x)
        initial_conv = self.initial_conv(x)
        rconv = initial_conv

        for _ in range(3):
            rconv = self.rconv(rconv)
        prelu = self.bn_prelu(rconv)

        add1 = initial_conv + prelu

        transpose_conv = self.transpose_conv(add1)
        conv1 = self.conv_128x64K3(transpose_conv)
        conv2 = self.conv_64xinK9(conv1)
        add2 = interp + conv2

        conv3 = self.conv_inx128K3(add2)
        conv4 = self.conv_128x64K3(conv3)
        conv5 = self.conv_64xinK9(conv4)
        add3 = add2 + conv5
        
        out = add3
        return out


