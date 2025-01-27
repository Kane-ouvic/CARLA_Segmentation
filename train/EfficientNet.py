import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block, used to enhance channel attention"""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    """Basic convolutional block, including SEBlock and convolutional layers + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_channels)  # 加入 SE 注意力機制

    def forward(self, x):
        x = self.conv(x)
        return self.se(x)

class UNetPlusPlus3(nn.Module):
    """改進的 UNet++ 架構，使用 EfficientNet-B3 編碼器和注意力機制"""
    def __init__(self, num_classes):
        super(UNetPlusPlus3, self).__init__()

        # 編碼器使用 EfficientNet-B3
        efficient_net = timm.create_model('efficientnet_b3', pretrained=True, features_only=True)
        self.encoder1 = efficient_net.feature_info[1]['module']  # 64 channels
        self.encoder2 = efficient_net.feature_info[2]['module']  # 176 channels
        self.encoder3 = efficient_net.feature_info[3]['module']  # 384 channels
        self.encoder4 = efficient_net.feature_info[4]['module']  # 1536 channels

        # 解碼器層 (具有密集跳躍連接和 Pyramid Pooling)
        self.upconv4 = nn.ConvTranspose2d(1536, 384, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(768, 384)  # 拼接後的輸入通道為 768
        
        self.upconv3 = nn.ConvTranspose2d(384, 176, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(352, 176)  # 拼接後的輸入通道為 352

        self.upconv2 = nn.ConvTranspose2d(176, 64, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(128, 64)   # 拼接後的輸入通道為 128

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(64, 32)    # 拼接後的輸入通道為 64

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # 編碼器
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # 解碼器和密集跳躍連接
        d4 = self.upconv4(e4)
        d4 = torch.cat((d4, e3), dim=1)
        d4 = self.conv4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.conv3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.conv2(d2)

        d1 = self.upconv1(d2)
        d1 = self.conv1(d1)

        out = self.final_conv(d1)
        out = F.interpolate(out, size=(600, 800), mode="bilinear", align_corners=False)
        return out