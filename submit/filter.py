import cv2
import os

def denoise_images(input_folder, output_folder):
    # 如果目標資料夾不存在，則建立資料夾
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍歷輸入資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        # 檢查是否為圖片檔案 (常見的副檔名)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # 讀取圖片
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            # 濾波處理 (這裡使用雙邊濾波，可根據需求更換)
            denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

            # 儲存濾波後的圖片到目標資料夾
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, denoised_img)
            print(f"{filename} 已完成處理並儲存到 {output_folder}")

# 使用範例
input_folder = "/home/ouvic/ML/ML_HW2/predict2/vote_1114_1"      # 輸入資料夾的路徑
output_folder = "/home/ouvic/ML/ML_HW2/predict2/vote_1114_1_filter"    # 輸出資料夾的路徑
denoise_images(input_folder, output_folder)
