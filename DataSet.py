import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 强力清洗函数 (保持不变)
# ---------------------------------------------------------
def clean_mask(mask, target_h, target_w):
    """
    无论掩码是 3D、多通道还是尺寸不对，都强制转为 (target_h, target_w) 的 2D 数组
    """
    if mask is None or mask.size == 0:
        return np.zeros((target_h, target_w), dtype=np.float32)

    if mask.ndim > 2:
        try:
            mask = np.max(mask, axis=-1)
        except Exception:
            mask = np.zeros((target_h, target_w), dtype=np.float32)

    if mask.ndim > 2: 
        mask = mask[:, :, 0]

    try:
        if target_h > 0 and target_w > 0:
            if mask.shape[:2] != (target_h, target_w):
                mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            return np.zeros((256, 256), dtype=np.float32)
    except Exception as e:
        # print(f"⚠️ Resize 失败: {e}, 重置为全黑。")
        return np.zeros((target_h, target_w), dtype=np.float32)

    return mask.astype(np.float32)

# ---------------------------------------------------------
# 2. Dataset 类
# ---------------------------------------------------------
class ForgeryDataset(Dataset):
    def __init__(self, image_paths, mask_dir, transform=None):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- A. 加载图像 ---
        img_path = self.image_paths[idx]
        file_name = os.path.basename(img_path)
        img_id = os.path.splitext(file_name)[0]
        
        default_h, default_w = 256, 256
        
        try:
            image = cv2.imread(img_path)
        except Exception:
            image = None

        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            # print(f"❌ [跳过坏图] ID: {img_id}")
            image = np.zeros((default_h, default_w, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # --- B. 加载掩码 ---
        mask_files = [f for f in os.listdir(self.mask_dir) if f.startswith(img_id) and f.endswith('.npy')]
        mask = None
        
        if mask_files:
            try:
                loaded_masks = []
                for f in mask_files:
                    m = np.load(os.path.join(self.mask_dir, f))
                    if m.shape == (): continue 
                    loaded_masks.append(m)
                
                if loaded_masks:
                    mask = np.logical_or.reduce(loaded_masks).astype(np.float32)
            except Exception:
                mask = None
        
        # --- C. 清洗掩码 ---
        mask = clean_mask(mask, h, w)

        # --- D. 增强与输出 ---
        if self.transform:
            try:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            except Exception:
                t = ToTensorV2()
                aug = t(image=image, mask=mask)
                image = aug['image']
                mask = aug['mask']
            
        # 确保转 Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)

        # ==========================================
        # 🛡️ 最终维度处理 
        # ==========================================
        # 1. 如果 mask 是 2D [H, W]，加上通道维度 -> [1, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        # 2. 如果 mask 是 3D [C, H, W] 且 C > 1 (多通道错误数据)，才进行压缩
        elif mask.ndim == 3 and mask.shape[0] > 1:
            mask = torch.max(mask, dim=0, keepdim=True)[0]

        # 此时 mask 必须是 [1, 256, 256]
        return image, mask, img_id

# ---------------------------------------------------------
# 3. 变换定义
# ---------------------------------------------------------
TRAIN_TRANSFORM = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], p=1.0)

# ---------------------------------------------------------
# 4. 可视化函数
# ---------------------------------------------------------
def visualize_sample(image, mask):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        image = np.ascontiguousarray(image)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
        if mask.ndim == 3: mask = mask[0]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    mask_colored = np.zeros_like(image, dtype=np.uint8)
    mask_colored[mask > 0.5] = [255, 0, 0]
    
    try:
        blended = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
        plt.imshow(blended)
    except Exception:
        plt.imshow(image)
        
    plt.title("Forgery Mask")
    plt.axis('off')

    plt.show()
