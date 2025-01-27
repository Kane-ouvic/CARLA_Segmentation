import os
import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

# 設定來源和目標資料夾
input_folder = "/home/ouvic/ML/ML_HW2/predict/vote_012_1102_2"
output_folder = "/home/ouvic/ML/ML_HW2/dialation/vote_1103_1"
os.makedirs(output_folder, exist_ok=True)

# 讀取每張 mask 圖片
for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # 讀取 mask 圖片
        mask_path = os.path.join(input_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 灰階讀取，假設每個像素表示類別標籤

        # 初始化空的 mask_probs 張量
        num_classes = 13  # 假設有 13 個類別
        height, width = mask.shape
        mask_probs = np.zeros((num_classes, height, width), dtype=np.float32)

        # 將 mask 轉為 one-hot 格式的概率張量
        for i in range(num_classes):
            mask_probs[i, :, :] = (mask == i).astype(np.float32)

        # 為了避免完全確定性，加入少量隨機噪聲
        mask_probs = mask_probs * 0.9 + 0.1 / num_classes

        # 構建 CRF 模型並設置 energy
        d = dcrf.DenseCRF2D(width, height, num_classes)
        unary = unary_from_softmax(mask_probs)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=cv2.imread(mask_path), compat=10)

        # 執行 CRF 優化
        Q = d.inference(5)
        result = np.argmax(Q, axis=0).reshape((height, width))

        # 儲存結果
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result)

print("Finished")
