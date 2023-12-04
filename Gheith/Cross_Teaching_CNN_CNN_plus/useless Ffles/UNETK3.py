class _ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p, kernel_size=3, padding=1):
        super(_ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv_conv(x)

class _DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p, kernel_size=3, padding=1):
        super(_DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _ConvBlock(in_channels, out_channels, dropout_p, kernel_size, padding=padding),
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class _UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(
        self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True, kernel_size=3, padding=1
    ):
        super(_UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2
            )
        self.conv = _ConvBlock(in_channels2 * 2, out_channels, dropout_p, kernel_size=kernel_size, padding=padding)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetEncoderK3(nn.Module):
    def __init__(self):
        super(UNetEncoderK3, self).__init__()
        self.in_conv = _ConvBlock(1, 16, 0.05)
        self.down1 = _DownBlock(16, 32, 0.1)
        self.down2 = _DownBlock(32, 64, 0.2)
        self.down3 = _DownBlock(64, 128, 0.3)
        self.down4 = _DownBlock(128, 256, 0.5)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        return [x0, x1, x2, x3, x4]


class UNetEncoderK5(nn.Module):
    def __init__(self):
        super(UNetEncoderK5, self).__init__()
        self.in_conv = _ConvBlock(1, 16, 0.05, 5, 2)
        self.down1 = _DownBlock(16, 32, 0.1, 5, 2)
        self.down2 = _DownBlock(32, 64, 0.2, 5, 2)
        self.down3 = _DownBlock(64, 128, 0.3, 5, 2)
        self.down4 = _DownBlock(128, 256, 0.5, 5, 2)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        return [x0, x1, x2, x3, x4]


class UNetDecoderK3(nn.Module):
    def __init__(self):
        super(UNetDecoderK3, self).__init__()
        self.up1 = _UpBlock(256, 128, 128, dropout_p=0.0)
        self.up2 = _UpBlock(128, 64, 64, dropout_p=0.0)
        self.up3 = _UpBlock(64, 32, 32, dropout_p=0.0)
        self.up4 = _UpBlock(32, 16, 16, dropout_p=0.0)

        self.out_conv = nn.Conv2d(16, 4, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output