import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg
from functools import partial

# -----------------------------------
# Transformer Block Implementation
# -----------------------------------

class Mlp(nn.Module):
    """Multi-Layer Perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping patches."""
    def __init__(self, img_size=768, patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class TransformerBlock(nn.Module):
    """Transformer Block with Multi-Head Self-Attention and MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, bias=qkv_bias)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        
    def forward(self, x):
        B, C, H, W = x.size()
        x_reshaped = x.flatten(2).permute(2, 0, 1)  # [N, B, C]
        x_norm = self.norm1(x_reshaped)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_reshaped + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(1, 2, 0).reshape(B, C, H, W)
        return x

# -----------------------------------
# MiT Encoder Implementation
# -----------------------------------

class MiTEncoder(nn.Module):
    """Mix Transformer (MiT) Encoder used in SegFormer."""
    def __init__(self, in_chans=3, embed_dims=[64, 128, 320, 512],
                 depths=[3, 4, 6, 3], num_heads=[1, 2, 5, 8]):
        super(MiTEncoder, self).__init__()
        self.embed_dims = embed_dims

        # Patch Embedding
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Transformer Encoder
        self.block1 = nn.ModuleList([
            TransformerBlock(dim=embed_dims[0], num_heads=num_heads[0]) for _ in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0], eps=1e-6)

        self.block2 = nn.ModuleList([
            TransformerBlock(dim=embed_dims[1], num_heads=num_heads[1]) for _ in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1], eps=1e-6)

        self.block3 = nn.ModuleList([
            TransformerBlock(dim=embed_dims[2], num_heads=num_heads[2]) for _ in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2], eps=1e-6)

        self.block4 = nn.ModuleList([
            TransformerBlock(dim=embed_dims[3], num_heads=num_heads[3]) for _ in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3], eps=1e-6)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stage 1
        x1 = self.patch_embed1(x)  # [B, C1, H1, W1]
        for blk in self.block1:
            x1 = blk(x1)
        x1 = self.norm1(x1.flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(x1)

        # Stage 2
        x2 = self.patch_embed2(x1)
        for blk in self.block2:
            x2 = blk(x2)
        x2 = self.norm2(x2.flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(x2)

        # Stage 3
        x3 = self.patch_embed3(x2)
        for blk in self.block3:
            x3 = blk(x3)
        x3 = self.norm3(x3.flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(x3)

        # Stage 4
        x4 = self.patch_embed4(x3)
        for blk in self.block4:
            x4 = blk(x4)
        x4 = self.norm4(x4.flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(x4)

        return [x1, x2, x3, x4]  # Return features from different stages

# -----------------------------------
# SegFormer Decoder Implementation
# -----------------------------------

class SegFormerDecoder(nn.Module):
    """SegFormer Decoder."""
    def __init__(self, embed_dims=[64, 128, 320, 512], num_classes=13):
        super(SegFormerDecoder, self).__init__()
        self.linear_c1 = nn.Conv2d(embed_dims[0], 256, kernel_size=1)
        self.linear_c2 = nn.Conv2d(embed_dims[1], 256, kernel_size=1)
        self.linear_c3 = nn.Conv2d(embed_dims[2], 256, kernel_size=1)
        self.linear_c4 = nn.Conv2d(embed_dims[3], 256, kernel_size=1)

        self.linear_fuse = nn.Conv2d(256 * 4, 256, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, features):
        x1, x2, x3, x4 = features

        x1 = self.linear_c1(x1)
        x1 = F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=False)

        x2 = self.linear_c2(x2)
        x2 = F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=False)

        x3 = self.linear_c3(x3)
        x3 = F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=False)

        x4 = self.linear_c4(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.linear_fuse(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_pred(x)
        x = F.interpolate(x, size=(600, 800), mode='bilinear', align_corners=False)
        return x

# -----------------------------------
# Complete SegFormer Model
# -----------------------------------

class SegFormer(nn.Module):
    """SegFormer architecture."""
    def __init__(self, num_classes=13, pretrained=False):
        super(SegFormer, self).__init__()
        self.encoder = MiTEncoder()
        self.decoder = SegFormerDecoder(num_classes=num_classes)

        if pretrained:
            self._load_pretrained_weights()

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out

    def _load_pretrained_weights(self):
        # Optionally, you can load pre-trained weights here
        pass

# -----------------------------------
# Example Usage
# -----------------------------------

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SegFormer(num_classes=13).to(device)

#     # For summary, we use a batch size of 1 and input size of (3, 768, 768)
#     summary(model, input_size=(1, 3, 768, 768))
