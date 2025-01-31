# ****[CARLA\_Segmentation](https://github.com/Kane-ouvic/CARLA_Segmentation)****

This project is focused on image segmentation using various deep learning models. The primary goal is to train, validate, and infer segmentation models on a given dataset, utilizing different architectures and techniques to improve performance.

---

## **Table of Contents**

* [Introduction]()
* [Features]()
* [Installation]()
* [Usage]()
* [Methods and Improvements]()
* [Results Analysis]()
* [File Structure]()
* [Example Image]()

---

## **Introduction**

This project implements different deep learning architectures for image segmentation on the **CARLA dataset**. The initial model was **U-Net**, but performance improvements led to adopting **U-Net++** and further enhancing it with **SEBlock, CRF, and ensemble learning** to achieve better results.

![Example Screenshot](https://github.com/Kane-ouvic/CARLA_Segmentation/blob/main/result/imgs/1.png)
![Example Screenshot](https://github.com/Kane-ouvic/CARLA_Segmentation/blob/main/result/masks/1.png)
![Example Screenshot](https://github.com/Kane-ouvic/CARLA_Segmentation/blob/main/result/imgs/51.png)
![Example Screenshot](https://github.com/Kane-ouvic/CARLA_Segmentation/blob/main/result/masks/51.png)
![Example Screenshot](https://github.com/Kane-ouvic/CARLA_Segmentation/blob/main/result/imgs/101.png)
![Example Screenshot](https://github.com/Kane-ouvic/CARLA_Segmentation/blob/main/result/masks/101.png)

## **Features**

* **Multiple segmentation models**: U-Net, U-Net++, and U-Net++ Improved
* **Advanced loss functions**: Cross-Entropy, Dice Loss, and Focal Loss
* **Data augmentation techniques**: Random hue adjustment, contrast adjustment
* **Post-processing**: Conditional Random Fields (CRF) to refine predictions
* **Ensemble learning**: Pixel voting mechanism to enhance accuracy

---

## **Installation**

Step-by-step guide on how to set up the project.

1. Clone the repository:
2. Install the required packages::
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

Provide examples or instructions on how to use the project.

Training

```bash
python main.py
```

Inference

```bash
python inference.py
```


## Methods and Improvements

To improve segmentation performance, the following techniques were applied:

1. **Model Architecture Enhancements:**
   * **Baseline:** U-Net
   * **Improved:** U-Net++
   * **Final Enhancement:** U-Net++ with SEBlock
2. **Encoder Progression:**
   * ResNet50 → ResNet101 → ResNet152 → ResNeXt101 (64x4d) → ResNeXt101 (32x8d)
3. **Loss Function Adjustments:**
   * Cross Entropy → Cross Entropy + Dice Loss → Focal Loss + Dice Loss
4. **Data Augmentation:**
   * Random hue adjustment, contrast adjustment
5. **Post-processing and Ensemble Learning:**
   * **CRF (Conditional Random Fields):** Used to reduce noise in predictions
   * **Ensemble Learning:** Combined multiple models' predictions via pixel voting

---

## Results Analysis

Training results for different configurations:

| Encoder                     | Epochs | Batch Size | Data Augmentation | Loss Function | Model Type | Resize | Test IoU |
| ----------------------------- | -------- | ------------ | ------------------- | --------------- | ------------ | -------- | ---------- |
| ResNet16                    | 200    | 16         | No                | Cross Entropy | U-Net      | 256    | 0.67434  |
| ResNet50                    | 300    | 16         | No                | Cross Entropy | U-Net      | 256    | 0.70367  |
| DenseNet201                 | 500    | 8          | No                | Dice + CE     | U-Net      | 256    | 0.67867  |
| ResNet50                    | 100    | 8          | No                | Dice + CE     | U-Net++    | 512    | 0.84004  |
| ResNet152                   | 200    | 8          | No                | Dice + CE     | U-Net++    | 512    | 0.85838  |
| ResNet152                   | 200    | 8          | Yes               | Dice + Focal  | U-Net++    | 512    | 0.85791  |
| ResNeXt101 (64x4d)          | 100    | 2          | No                | Dice + Focal  | U-Net++I   | 768    | 0.89737  |
| ResNeXt101 (32x8d)          | 100    | 2          | No                | Dice + Focal  | U-Net++I   | 768    | 0.89604  |
| Ensemble (U-Net++ Improved) | -      | -          | -                 | -             | -          | -      | 0.90666  |

### Key Findings:

* **Larger models perform better** with **higher resolution images**.
* **Loss function choice has limited impact**, possibly due to the large dataset size (4000+ images).
* **CRF and ensemble techniques provided marginal improvements** to IoU.

---

## **File Structure**

Describe the project's directory and file layout.

```plaintext
CARLA_Segmentation/
├── fig/               #  Graphs of the training process
├── log/              # Logs of the training process
├── result/          #  Inference results
├── submit/        # Code uploaded to Kaggle
├── train/           # Training code. Include model structures, Loss function.
```

---

### Example Image

![Example Screenshot](https://github.com/Kane-ouvic/CARLA_Segmentation/blob/main/result/imgs/1.png)

