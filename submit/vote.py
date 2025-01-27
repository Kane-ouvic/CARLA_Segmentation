import os
import numpy as np
from PIL import Image

# 定義三個資料夾的路徑
# folder1 = "/home/ouvic/ML/ML_HW2/predict2/pred_1103_1"
# folder2 = "/home/ouvic/ML/ML_HW2/predict2/pred_UnetPlusPlus_1103"
# folder3 = "/home/ouvic/ML/ML_HW2/predict2/pred_UnetPlusPlusImproved_1104"
# folder4 = "/home/ouvic/ML/ML_HW2/predict2/pred_1109_1"
# folder5 = "/home/ouvic/ML/ML_HW2/predict2/pred_1110_1"
# folder6 = "/home/ouvic/ML/ML_HW2/predict2/pred_UnetPlusPlusImproved_1109"
# folder7 = "/home/ouvic/ML/ML_HW2/predict2/pred_UnetPlusPlusImproved_1110"

# 定義資料夾路徑的陣列
# model1_dirs = [f"/home/ouvic/ML/ML_HW2/predict3/pred_1116_model1_{i}" for i in range(1, 51)]
# model2_dirs = [f"/home/ouvic/ML/ML_HW2/predict3/pred_1116_model2_{i}" for i in range(1, 51)]
# folders = [
#     "/home/ouvic/ML/ML_HW2/predict2/vote_1106_1",
#     "/home/ouvic/ML/ML_HW2/predict2/vote_1112_1",
#     "/home/ouvic/ML/ML_HW2/predict2/vote_1114_1"
# ] + model1_dirs

folders = [
    "/home/ouvic/ML/ML_HW2/final_predict/vote_1119_x3_3",
    # "/home/ouvic/ML/ML_HW2/predict2/vote_1114_2",
    # "/home/ouvic/ML/ML_HW2/predict2/pred_1119_final_1",
    "/home/ouvic/ML/ML_HW2/predict2/pred_1122_final_1",
    "/home/ouvic/ML/ML_HW2/predict2/pred_1122_final_2",
    "/home/ouvic/ML/ML_HW2/predict2/pred_1122_final_3",
    "/home/ouvic/ML/ML_HW2/predict2/vote_1114_1"
] 

# 定義輸出結果的資料夾
output_folder = f"/home/ouvic/ML/ML_HW2/final_predict/vote_1122_x{len(folders)}_3"
os.makedirs(output_folder, exist_ok=True)

# 取得三個資料夾中的圖片名稱列表（假設三個資料夾的圖片名稱一致）
image_names = os.listdir(folders[0])

# 對每張圖片進行投票整合
for image_name in image_names:
    
    images = []
    # 讀取每個資料夾中對應的圖片
    for folder in folders:
        image_path = os.path.join(folder, image_name)
        image_data = np.array(Image.open(image_path))
        images.append(image_data)
    # image4 = np.array(Image.open(os.path.join(folder4, image_name)))
    # image5 = np.array(Image.open(os.path.join(folder5, image_name)))
    # image6 = np.array(Image.open(os.path.join(folder6, image_name)))
    # image7 = np.array(Image.open(os.path.join(folder7, image_name)))
    
    # 將所有圖片堆疊起來，形成形狀為 (H, W, len(folders)) 的陣列
    stacked_images = np.stack(images, axis=-1)
    
    # 使用投票機制對每個像素選擇出現最多的類別
    final_image = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=-1, arr=stacked_images)
    
    # 將結果轉換為圖片格式並保存
    final_image = Image.fromarray(final_image.astype(np.uint8))
    final_image.save(os.path.join(output_folder, image_name))