import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
from torchsummary import summary
import timm


class SimpleUNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleUNet, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(600, 800), mode="bilinear", align_corners=False)  # 上採樣到原尺寸
        return x
    

class UNet_2048(nn.Module):
    def __init__(self, num_classes, encoder_name="resnet18"):
        super(UNet_2048, self).__init__()
        self.encoder = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)  # 直接輸出類別通道數
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(600, 800), mode="bilinear", align_corners=False)  # 上採樣到原尺寸
        return x


class UNet_2048_2(nn.Module):
    def __init__(self, num_classes, encoder_name="resnet18"):
        super(UNet_2048_2, self).__init__()
        self.encoder = models.densenet201(pretrained=True)
        # 確保 encoder 的輸出通道數是 1920
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        # 定義解碼器 (保持不變)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1920, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)  # 直接輸出類別通道數
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(600, 800), mode="bilinear", align_corners=False)  # 上採樣到原尺寸
        return x