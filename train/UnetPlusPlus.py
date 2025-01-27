import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
from torchsummary import summary
import timm


class ConvBlock(nn.Module):
    """Basic convolutional block, including two convolutional layers + BatchNorm + ReLU"""
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

    def forward(self, x):
        return self.conv(x)
    

class UNetPlusPlus(nn.Module):
    """UNet++ architecture"""
    def __init__(self, num_classes, encoder_name='resnext101_32x8d'):
        super(UNetPlusPlus, self).__init__()
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
        else:
            raise ValueError(f"Unsupported encoder name: {encoder_name}")
        self.encoder1 = nn.Sequential(*list(resnet.children())[:3])   # Output channels: 64
        self.encoder2 = nn.Sequential(*list(resnet.children())[3:5])  # Output channels: 256
        self.encoder3 = nn.Sequential(*list(resnet.children())[5])    # Output channels: 512
        self.encoder4 = nn.Sequential(*list(resnet.children())[6])    # Output channels: 1024
        self.encoder5 = nn.Sequential(*list(resnet.children())[7])    # Output channels: 2048

        # Decoder layers (with dense skip connections)
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(2048, 1024) # Modify input channels to 2048

        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(1024, 512)  # Modify input channels to 1024

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(512, 256)  # Modify input channels to 512

        self.upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(128, 64)  # Modify input channels to 128

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # Output channels: 64
        e2 = self.encoder2(e1)  # Output channels: 256
        e3 = self.encoder3(e2)  # Output channels: 512
        e4 = self.encoder4(e3)  # Output channels: 1024
        e5 = self.encoder5(e4)  # Output channels: 2048

        # Decoder and dense skip connections
        d4 = self.upconv4(e5)   # Output channels: 1024
        d4 = torch.cat((d4, e4), dim=1)  # Concatenation channels: 1024 + 1024 = 2048
        d4 = self.conv4(d4)     # Output channels: 1024

        d3 = self.upconv3(d4)   # Output channels: 512
        d3 = torch.cat((d3, e3), dim=1)  # Concatenation channels: 512 + 512 = 1024
        d3 = self.conv3(d3)     # Output channels: 512

        d2 = self.upconv2(d3)   # Output channels: 256
        d2 = torch.cat((d2, e2), dim=1)  # Concatenation channels: 256 + 256 = 512
        d2 = self.conv2(d2)     # Output channels: 256

        d1 = self.upconv1(d2)   # Output channels: 64
        d1 = torch.cat((d1, e1), dim=1)  # Concatenation channels: 64 + 64 = 128
        d1 = self.conv1(d1)     # Output channels: 64

        out = self.final_conv(d1)
        out = nn.functional.interpolate(out, size=(600, 800), mode="bilinear", align_corners=False)
        return out
    
class UNetPlusPlus2(nn.Module):
    def __init__(self, num_classes):
        super(UNetPlusPlus2, self).__init__()

        self.encoder = timm.create_model('swin_base_patch4_window7_224', pretrained=True, features_only=True)
        
        self.encoder_layers = [
            self.encoder.stem,          # Stage 0
            self.encoder.layers[0],     # Stage 1
            self.encoder.layers[1],     # Stage 2
            self.encoder.layers[2],     # Stage 3
            self.encoder.layers[3],     # Stage 4
        ]
        
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        features = []
        out = x
        for layer in self.encoder_layers:
            out = layer(out)
            features.append(out)
        
        e1 = features[0]  # 128
        e2 = features[1]  # 256
        e3 = features[2]  # 512
        e4 = features[3]  # 1024
        e5 = features[4]  # 1024
        
        d4 = self.upconv4(e5)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.conv4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.conv3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.conv2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.conv1(d1)

        out = self.final_conv(d1)
        out = nn.functional.interpolate(out, size=(600, 800), mode="bilinear", align_corners=False)
        return out

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNetPlusPlus(num_classes=13, encoder_name='resnext101_32x8d').to(device)
# summary(model, (3, 512, 512))