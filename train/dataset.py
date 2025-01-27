import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 自定義資料集類別
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)  # 讀取為灰階，mask 為0~12的類別編碼
        # mask = (mask == 255).astype('uint8')  # 將 255 映射為 1，保持 0 不變

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert mask to tensor and keep it as long
        mask = torch.tensor(mask, dtype=torch.long)
        
        return image, mask

class AugmentedSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augment_factor=3, augment_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment_factor = augment_factor
        self.augment_transform = augment_transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images) * (1 + self.augment_factor)

    def __getitem__(self, idx):
        image_idx = idx // (1 + self.augment_factor)
        img_path = os.path.join(self.image_dir, self.images[image_idx])
        mask_path = os.path.join(self.mask_dir, self.masks[image_idx])

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        mask = torch.tensor(mask, dtype=torch.long)

        if idx % (1 + self.augment_factor) == 0:
            # Return original image and mask
            if self.transform:
                image = self.transform(image)
            return image, mask
        else:
            # Return augmented image and mask
            if self.augment_transform:
                augmented_image = self.augment_transform(image)
            return augmented_image, mask


def prepare_dataloaders(image_dir, mask_dir, batch_size=16, val_ratio=0.1, augment_factor=3, transform=None, augment_transform=None):
    # 加載資料集
    # dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)。
    dataset = AugmentedSegmentationDataset(image_dir, mask_dir, transform=transform, augment_factor=augment_factor, augment_transform=augment_transform)

    # 計算訓練集和驗證集的大小
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    # 分割資料集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 創建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader