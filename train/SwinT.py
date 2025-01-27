import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchsummary import summary

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

class SwinUNetPlusPlus(nn.Module):
    """UNet++ architecture with Swin Transformer encoder"""
    def __init__(self, num_classes, encoder_name='swin_base_patch4_window7_224', img_size=768):
        super(SwinUNetPlusPlus, self).__init__()

        # Load pre-trained Swin Transformer encoder with specified image size
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=True,
            img_size=img_size,
            features_only=True
        )

        # Get actual encoder channels
        encoder_channels = self.encoder.feature_info.channels()
        print(f"Encoder channels: {encoder_channels}")

        decoder_channels = [512, 256, 128, 64]

        # Decoder layers
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(encoder_channels[3], decoder_channels[0], kernel_size=1)
        )
        self.conv3 = ConvBlock(encoder_channels[2] + decoder_channels[0], decoder_channels[0])

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(decoder_channels[0], decoder_channels[1], kernel_size=1)
        )
        self.conv2 = ConvBlock(encoder_channels[1] + decoder_channels[1], decoder_channels[1])

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(decoder_channels[1], decoder_channels[2], kernel_size=1)
        )
        self.conv1 = ConvBlock(encoder_channels[0] + decoder_channels[2], decoder_channels[2])

        self.upconv0 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(decoder_channels[2], decoder_channels[3], kernel_size=1)
        )
        self.conv0 = ConvBlock(decoder_channels[3], decoder_channels[3])

        self.final_conv = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
        
    def forward(self, x):
        # Swin Transformer Encoder
        encoder_outputs = self.encoder(x)

        c1 = encoder_outputs[0]  # Stage 1 output
        c2 = encoder_outputs[1]  # Stage 2 output
        c3 = encoder_outputs[2]  # Stage 3 output
        c4 = encoder_outputs[3]  # Stage 4 output

        # Decoder with skip connections
        d3 = self.upconv3(c4)  # Up-sample
        d3 = torch.cat([d3, c3], dim=1)
        d3 = self.conv3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, c2], dim=1)
        d2 = self.conv2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, c1], dim=1)
        d1 = self.conv1(d1)

        d0 = self.upconv0(d1)
        d0 = self.conv0(d0)

        out = self.final_conv(d0)
        out = F.interpolate(out, size=(600, 800), mode="bilinear", align_corners=False)
        return out

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SwinUNetPlusPlus(
#     num_classes=13,
#     encoder_name='swin_base_patch4_window12_224',
#     img_size=384
# ).to(device)
# summary(model, (3, 384, 384))