import cv2
import os
import numpy as np
from collections import Counter

def calculate_pixel_distribution(folder_path):
    # 初始化像素值分布
    pixel_distribution = np.zeros(256, dtype=int)

    # 遍歷資料夾內所有檔案
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 確保是圖片檔案
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 讀取圖片
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # 計算當前圖片中每個像素值的數量
                unique, counts = np.unique(image, return_counts=True)
                # 將結果累加到 pixel_distribution
                for value, count in zip(unique, counts):
                    pixel_distribution[value] += count

    # 輸出結果
    for value, count in enumerate(pixel_distribution):
        print(f"像素值 {value}: {count} 個")

    return pixel_distribution

# 設定資料夾路徑
folder_path = "/home/ouvic/ML/ML_HW2/final_predict/vote_1119_x3_2"  # 替換成你的資料夾路徑
pixel_distribution = calculate_pixel_distribution(folder_path)