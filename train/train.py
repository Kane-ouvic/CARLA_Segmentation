import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=device, num_classes=13):
    train_loss_history = []
    val_loss_history = []
    train_iou_history = []
    val_iou_history = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # 計算 IoU
            batch_iou = calculate_iou(outputs, masks, num_classes)
            running_iou += batch_iou

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_iou = running_iou / len(train_loader)
        train_loss_history.append(epoch_train_loss)
        train_iou_history.append(epoch_train_iou)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_iou = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, masks)
                val_running_loss += val_loss.item()

                # 計算 IoU
                batch_iou = calculate_iou(outputs, masks, num_classes)
                val_running_iou += batch_iou

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_iou = val_running_iou / len(val_loader)
        val_loss_history.append(epoch_val_loss)
        val_iou_history.append(epoch_val_iou)
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"./pth/UnetPlusPlus_64x4d_1118_{epoch+1}.pth")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}, Train IoU: {epoch_train_iou}, Val IoU: {epoch_val_iou}")

    torch.save(model.state_dict(), "./pth/UnetPlusPlus_64x4d_1118.pth")
    print("Model weights saved")

    # 繪製 Training 和 Validation Loss 圖
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_loss_history, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_loss_history, label="Validation Loss", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.savefig('./loss.png')
    
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_iou_history, label="Training IoU")
    plt.plot(range(1, num_epochs + 1), val_iou_history, label="Validation IoU", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Training and Validation IoU over Epochs")
    plt.legend()
    plt.savefig('./iou.png')
    

def inference(model, image_dir, output_dir, transform, device, num_classes):
    model.eval()  # 設置模型為評估模式

    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():  # 關閉梯度計算
        # print(os.listdir(image_dir))
        for image_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, image_name)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Unable to read image '{img_path}'. Skipping this file.")
                continue
            
            # 進行影像轉換
            input_image = transform(image).unsqueeze(0).to(device)  # 增加 batch 維度
            
            # 模型推理
            output = model(input_image)
            output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # 取每個像素的最大值索引作為類別
            
            # 保存或顯示結果
            output_img = Image.fromarray((output * (255 // (num_classes - 1))).astype(np.uint8))  # 可視化結果
            output_img.save(os.path.join(output_dir, f"{image_name}"))
            # print(f"Processed {image_name}")
            
            
def inferenceCRF(model, image_dir, output_dir, transform, device, num_classes):
    model.eval()  # 設置模型為評估模式

    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():  # 關閉梯度計算
        for image_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, image_name)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Unable to read image '{img_path}'. Skipping this file.")
                continue
            
            # 進行影像轉換
            input_image = transform(image).unsqueeze(0).to(device)  # 增加 batch 維度
            
            # 模型推理
            output = model(input_image)
            output_probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()  # 取得 softmax 概率
            
            # 構建 CRF
            height, width = image.shape[:2]
            d = dcrf.DenseCRF2D(width, height, num_classes)

            # 設定 unary 預測概率
            unary = unary_from_softmax(output_probs)
            d.setUnaryEnergy(unary)

            # 加入 pairwise 參數來保持邊緣
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

            # 執行 CRF 推理
            Q = d.inference(5)
            crf_output = np.argmax(Q, axis=0).reshape((height, width))

            # 保存或顯示結果
            output_img = Image.fromarray((crf_output * (255 // (num_classes - 1))).astype(np.uint8))  # 可視化結果
            output_img.save(os.path.join(output_dir, f"{image_name}"))
            print(f"Processed {image_name} with CRF")


def calculate_iou(outputs, masks, num_classes):
    # 將模型輸出轉換為類別預測
    outputs = torch.argmax(outputs, dim=1)
    ious = []
    for cls in range(1, num_classes):  # 假設類別 0 是背景
        pred = (outputs == cls)
        target = (masks == cls)
        intersection = (pred & target).sum().item()
        union = (pred | target).sum().item()
        if union == 0:
            iou = float('nan')  # 如果沒有該類別的預測或標註，忽略該類別
        else:
            iou = intersection / union
            ious.append(iou)
    if len(ious) == 0:
        return 0.0
    else:
        return np.nanmean(ious)  # 計算平均 IoU，忽略 NaN

