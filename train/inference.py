import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SegmentationDataset, prepare_dataloaders
from utils import CombinedLoss
from UnetPlusPlus import UNetPlusPlus
from UnetPlusPlusImproved import UNetPlusPlusImproved
from train import train_model, inference, inferenceCRF
import os


if __name__ == "__main__":
    # 路徑和設備設置
    inference_image_dir = "/home/ouvic/ML/ML_HW2/hw2_dataset/test/imgs"  # 推理影像資料夾
    output_dir = "/home/ouvic/ML/ML_HW2/predict2/pred_1122_final_3"  # 儲存推理結果的資料夾
    
    # output_dirs = [f"/home/ouvic/ML/ML_HW2/predict3/pred_1116_model2_{i}" for i in range(11, 51)]
    checkpoint_path = "/home/ouvic/ML/ML_HW2/project/pth/UnetPlusPlus_64x4d_1118_80.pth"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # 資料增強和轉換
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((768, 768)),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    
    # 設定超參數
    num_classes = 13
    # 資料加載
    # 模型、損失函數和優化器
    # model = SimpleUNet(num_classes=num_classes).to(device)
    # model = UNet_2048_2(num_classes=num_classes).to(device)
    # model = UNetPlusPlus(num_classes=num_classes, encoder_name='resnext101_64x4d').to(device)
    # model = nn.DataParallel(UNetPlusPlusImproved(num_classes=num_classes, encoder_name='resnext101_32x8d')).to(device)
    model = UNetPlusPlusImproved(num_classes=num_classes, encoder_name='resnext101_64x4d').to(device)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model weights from {checkpoint_path}")
    # 執行推理
    # for output_dir in output_dirs:
    #     inference(model, inference_image_dir, output_dir, transform, device, num_classes)
    inference(model, inference_image_dir, output_dir, transform, device, num_classes)
    # inferenceCRF(model, inference_image_dir, output_dir, transform, device, num_classes)
