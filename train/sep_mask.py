import os
import numpy as np
from PIL import Image

# 定義原始 mask 資料夾路徑和輸出資料夾路徑
mask_folder = "/home/ouvic/ML/ML_HW2/test_dataset/train/masks"  # 原始 mask 資料夾
output_base_folder = "/home/ouvic/ML/ML_HW2/test_dataset/train"  # 輸出資料夾基礎路徑

# 創建每個標籤的資料夾（1到12）
for label in range(1, 13):
    os.makedirs(os.path.join(output_base_folder, f"class_{label}"), exist_ok=True)

# 遍歷 mask 資料夾中的所有圖片
for mask_name in os.listdir(mask_folder):
    mask_path = os.path.join(mask_folder, mask_name)
    mask = np.array(Image.open(mask_path))  # 讀取圖片並轉換為 numpy 陣列

    # 對每個標籤生成單獨的掩碼
    for label in range(1, 13):
        # 創建一個空白掩碼，初始值為 0
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        
        # 將該類別的像素設為 255，其他保持為 0
        binary_mask[mask == label] = 255
        
        # 保存該類別的掩碼圖片
        output_path = os.path.join(output_base_folder, f"class_{label}", mask_name)
        Image.fromarray(binary_mask).save(output_path)

print("Finished")
