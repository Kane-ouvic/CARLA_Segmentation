import cv2
import os
import numpy as np

# 定義資料夾路徑
input_dir = '/home/ouvic/ML/ML_HW2/final_predict/vote_1122_x5_3'  # 來源影像資料夾
output_dir = '/home/ouvic/ML/ML_HW2/final_predict/vote_012_1122_x5_3'  # 儲存結果的資料夾

# 創建輸出資料夾（若不存在）
os.makedirs(output_dir, exist_ok=True)

# 設定原始像素值與目標值之間的映射（假設已有 13 個灰度值）
original_values = [0, 21, 42, 63, 84,105, 126, 147, 168, 189, 210, 231, 252]
new_values = list(range(len(original_values)))  # 產生 0~12 的對應值

# 創建查找表
lookup_table = np.zeros(256, dtype=np.uint8)
for orig_val, new_val in zip(original_values, new_values):
    lookup_table[orig_val] = new_val

# 遍歷資料夾中的所有圖像
for image_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, image_name)
    
    # 讀取圖像
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Warning: Unable to read image '{input_path}'. Skipping this file.")
        continue
    
    # 應用查找表來重新映射圖像像素
    remapped_image = cv2.LUT(image, lookup_table)
    
    # 儲存重新映射後的圖像
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, remapped_image)
    
    print(f"Processed {image_name}")
