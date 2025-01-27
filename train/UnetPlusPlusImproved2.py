import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from resnext import resnext152

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
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

class SCSEBlock(nn.Module):
    """Spatial and Channel Squeeze & Excitation Block"""
    def __init__(self, in_channels, reduction=16):
        super(SCSEBlock, self).__init__()
        # Channel Squeeze and Excitation
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Spatial Squeeze and Excitation
        self.sse = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        cse = x * self.cse(x)
        sse = x * self.sse(x)
        return cse + sse

class AttentionGate(nn.Module):
    """Attention Gate"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=8, num_channels=F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=8, num_channels=F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_scse=True):
        super(ConvBlock, self).__init__()
        self.use_scse = use_scse
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        if use_scse:
            self.scse = SCSEBlock(out_channels)
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual = nn.Identity()
        self.dropout = nn.Dropout2d(p=0.5)
    def forward(self, x):
        residual = self.residual(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.use_scse:
            out = self.scse(out)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out

class UNetPlusPlusImproved2(nn.Module):
    """Improved UNet++ architecture"""
    def __init__(self, num_classes, encoder_name='resnext101_64x4d'):
        super(UNetPlusPlusImproved2, self).__init__()

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

        self.encoder1 = nn.Sequential(*list(resnet.children())[:3])   # Output channel: 64
        self.encoder2 = nn.Sequential(*list(resnet.children())[3:5])  # Output channel: 256
        self.encoder3 = nn.Sequential(*list(resnet.children())[5])    # Output channel: 512
        self.encoder4 = nn.Sequential(*list(resnet.children())[6])    # Output channel: 1024
        self.encoder5 = nn.Sequential(*list(resnet.children())[7])    # Output channel: 2048

        # Attention Gates
        self.attention_gate4 = AttentionGate(F_g=1024, F_l=1024, F_int=512)
        self.attention_gate3 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.attention_gate2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.attention_gate1 = AttentionGate(F_g=64, F_l=64, F_int=32)

        # Decoder layers (using interpolation upsampling and SCSE module)
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(2048, 1024, kernel_size=1)
        )
        self.conv4 = ConvBlock(2048, 1024, use_scse=True)

        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, kernel_size=1)
        )
        self.conv3 = ConvBlock(1024, 512, use_scse=True)

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=1)
        )
        self.conv2 = ConvBlock(512, 256, use_scse=True)

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 64, kernel_size=1)
        )
        self.conv1 = ConvBlock(128, 64, use_scse=True)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
    # 编码器
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # 解码器和密集跳跃连接
        d4 = self.upconv4(e5)
        e4_att = self.attention_gate4(d4, e4)
        d4 = torch.cat((d4, e4_att), dim=1)
        d4 = self.conv4(d4)

        d3 = self.upconv3(d4)
        e3_att = self.attention_gate3(d3, e3)
        d3 = torch.cat((d3, e3_att), dim=1)
        d3 = self.conv3(d3)

        d2 = self.upconv2(d3)
        e2_att = self.attention_gate2(d2, e2)
        d2 = torch.cat((d2, e2_att), dim=1)
        d2 = self.conv2(d2)

        d1 = self.upconv1(d2)
        e1_att = self.attention_gate1(d1, e1)
        d1 = torch.cat((d1, e1_att), dim=1)
        d1 = self.conv1(d1)

        out = self.final_conv(d1)
        out = nn.functional.interpolate(out, size=(600, 800), mode="bilinear", align_corners=False)
        return out

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetPlusPlusImproved2(num_classes=13, encoder_name='resnext101_32x8d').to(device)
summary(model, (3, 768, 768))