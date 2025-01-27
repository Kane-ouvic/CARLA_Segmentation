import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SegmentationDataset, prepare_dataloaders
from utils import CombinedLoss, CombinedLoss2
from Unet import SimpleUNet, UNet_2048, UNet_2048_2 
from UnetPlusPlus import UNetPlusPlus, UNetPlusPlus2
from UnetPlusPlusImproved import UNetPlusPlusImproved
from UnetPlusPlusImproved2 import UNetPlusPlusImproved2
from SegFormer import SegFormer
from EfficientNet import UNetPlusPlus3
from train import train_model, inference
from random import seed
import segmentation_models_pytorch as smp
import os

if __name__ == "__main__":
    # 路徑和設備設置
    image_dir = "/home/ouvic/ML/ML_HW2/hw2_dataset/train/imgs"  # 訓練影像資料夾路徑
    mask_dir = "/home/ouvic/ML/ML_HW2/hw2_dataset/train/masks"    # Mask資料夾路徑
    inference_image_dir = "/home/ouvic/ML/ML_HW2/hw2_dataset/test/imgs"  # 推理影像資料夾
    output_dir = "/home/ouvic/ML/ML_HW2/predict2/pred_UnetPlusPlus_1118"  # 儲存推理結果的資料夾
    checkpoint_path = "/home/ouvic/ML/ML_HW2/project/pth/UnetPlusPlusImproved_1103.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 資料增強和轉換
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((768, 768)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    augment_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((768, 768)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 隨機調整顏色
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    
    # 設定超參數
    num_classes = 13
    num_epochs = 100
    learning_rate = 1e-4
    batch_size = 3

    # 資料加載
    # dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
    # dataloader = DataLoader(dataset, batch_size, shuffle=True)
    train_loader, val_loader = prepare_dataloaders(image_dir, mask_dir, batch_size=batch_size, val_ratio=0.01, transform=transform, augment_factor=2, augment_transform=augment_transform)
    # train_loader, val_loader = prepare_dataloaders2(image_dir, mask_dir, batch_size=batch_size, val_ratio=0.05)
    torch.cuda.manual_seed(42)
    
    # 模型、損失函數和優化器
    # model = SimpleUNet(num_classes=num_classes).to(device)
    # model = UNet_2048_2(num_classes=num_classes).to(device)
    # model = UNetPlusPlus(num_classes=num_classes, encoder_name='resnext101_64x4d').to(device)
    model = UNetPlusPlusImproved(num_classes=num_classes, encoder_name='resnext101_64x4d').to(device)
    # model = UNetPlusPlusImproved2(num_classes=num_classes, encoder_name='resnext101_32x8d').to(device)
    # model = SegFormer(num_classes=num_classes).to(device)
    # criterion = smp.losses.DiceLoss(mode='multiclass')
    criterion = CombinedLoss2(weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model weights from {checkpoint_path}")

    # 訓練模型
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    # 執行推理
    inference(model, inference_image_dir, output_dir, transform, device, num_classes)
