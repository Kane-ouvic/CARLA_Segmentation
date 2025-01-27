import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from resnext import resnext152


class SEBlock(nn.Module):
    """Squeeze-and-Excitation"""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Global Average Pooling
        y = x.view(batch_size, channels, -1).mean(dim=2)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)
        )
    
    def forward(self, x):
        return self.conv(x)
    

class UNetPlusPlusImproved(nn.Module):
    """Improved UNet++ architecture"""
    def __init__(self, num_classes, encoder_name='resnext101_64x4d'):
        super(UNetPlusPlusImproved, self).__init__()

        # Pre-trained ResNet encoder
        if encoder_name == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        elif encoder_name == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        elif encoder_name == 'resnet152':
            resnet = models.resnet152(pretrained=True)
        elif encoder_name == 'resnext101_64x4d':
            resnet = models.resnext101_64x4d(pretrained=True)
        elif encoder_name == 'resnext101_32x8d':
            resnet = models.resnext101_32x8d(pretrained=True)
        elif encoder_name == 'resnext152':
            resnet = resnext152()
        else:
            raise ValueError(f"Unsupported encoder name: {encoder_name}")
        self.encoder1 = nn.Sequential(*list(resnet.children())[:3])   # Output channel: 64
        self.encoder2 = nn.Sequential(*list(resnet.children())[3:5])  # Output channel: 256
        self.encoder3 = nn.Sequential(*list(resnet.children())[5])    # Output channel: 512
        self.encoder4 = nn.Sequential(*list(resnet.children())[6])    # Output channel: 1024
        self.encoder5 = nn.Sequential(*list(resnet.children())[7])    # Output channel: 2048

        # Decoder layers (using interpolation upsampling and SE module)
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(2048, 1024, kernel_size=1)
        )
        self.conv4 = ConvBlock(2048, 1024)

        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, kernel_size=1)
        )
        self.conv3 = ConvBlock(1024, 512)

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=1)
        )
        self.conv2 = ConvBlock(512, 256)

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 64, kernel_size=1)
        )
        self.conv1 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.encoder1(x)    # Output channel: 64
        e2 = self.encoder2(e1)   # Output channel: 256
        e3 = self.encoder3(e2)   # Output channel: 512
        e4 = self.encoder4(e3)   # Output channel: 1024
        e5 = self.encoder5(e4)   # Output channel: 2048

        # 解码器和密集跳跃连接
        d4 = self.upconv4(e5)    # Output channel: 1024
        d4 = torch.cat((d4, e4), dim=1)  # Output channel: 1024 + 1024 = 2048
        d4 = self.conv4(d4)      # Output channel: 1024

        d3 = self.upconv3(d4)    # Output channel: 512
        d3 = torch.cat((d3, e3), dim=1)  # Output channel: 512 + 512 = 1024
        d3 = self.conv3(d3)      # Output channel: 512

        d2 = self.upconv2(d3)    # Output channel: 256
        d2 = torch.cat((d2, e2), dim=1)  # Output channel: 256 + 256 = 512
        d2 = self.conv2(d2)      # Output channel: 256

        d1 = self.upconv1(d2)    # Output channel: 64
        d1 = torch.cat((d1, e1), dim=1)  # Output channel: 64 + 64 = 128
        d1 = self.conv1(d1)      # Output channel: 64

        out = self.final_conv(d1)
        out = nn.functional.interpolate(out, size=(600, 800), mode="bilinear", align_corners=False)
        return out
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNetPlusPlusImproved(num_classes=13, encoder_name='resnext101_64x4d').to(device)
# summary(model, (3, 768, 768))