import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# 定義 Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        smooth = 1e-5
        inputs = nn.functional.softmax(inputs, dim=1)
        total_loss = 0
        for i in range(self.num_classes):
            input_flat = inputs[:, i].contiguous().view(-1)
            target_flat = (targets == i).float().view(-1)
            intersection = (input_flat * target_flat).sum()
            union = input_flat.sum() + target_flat.sum()
            loss = 1 - ((2. * intersection + smooth) / (union + smooth))
            total_loss += loss
        return total_loss / self.num_classes

# 定義 Combined Loss: dice loss 和 cross entropy loss 的加權平均
class CombinedLoss(nn.Module):
    def __init__(self, num_classes, weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.weight = weight
    
    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.weight * ce + (1 - self.weight) * dice

# 定義 Combined Loss2: dice loss 和 focal loss 的加權平均
class CombinedLoss2(nn.Module):
    def __init__(self, weight=0.5):
        super(CombinedLoss2, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass')
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass')
        self.weight = weight
    
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.weight * focal + (1 - self.weight) * dice