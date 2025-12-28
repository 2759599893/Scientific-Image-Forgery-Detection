import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import random

# =============================================================
# 1. 🛠️ 配置区域 (请修改这里!)
# =============================================================
# 新模型的权重文件路径 (确保是那个 efficientnet-b4 的模型)
MODEL_PATH = "D:/InfSec/best_checkpoint.pth4.0.tar" 

# 数据集根目录
DATA_ROOT = r"D:\InfSec\Data\recodai-luc-scientific-image-forgery-detection" 

# 测试图片目录 (建议指向 supplemental_images 看看新数据效果)
#TEST_IMG_DIR = os.path.join(DATA_ROOT, "supplemental_images") 
TEST_IMG_DIR = os.path.join(DATA_ROOT, "train_images/forged") # 也可以测原来的

# 掩码目录 (要对应上面的图片目录)
#MASK_DIR = os.path.join(DATA_ROOT, "supplemental_masks")
MASK_DIR = os.path.join(DATA_ROOT, "train_masks")

# 推理分辨率 (必须与训练时一致!!)
INPUT_SIZE = 512 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================
# 2. 🧠 模型构建 (升级为 EfficientNet-B4)
# =============================================================
def build_model():
    print(f"🏗️ 正在构建模型: EfficientNet-B4 (输入分辨率 {INPUT_SIZE}x{INPUT_SIZE})...")
    return smp.Unet(
        encoder_name="efficientnet-b4", # ⚠️ 必须匹配训练时的 Encoder
        encoder_weights=None,           # 推理模式不需要下载预训练权重
        in_channels=3,                  # 纯 RGB
        classes=1, 
        activation=None
    )

def load_checkpoint(path, device):
    if not os.path.exists(path):
        print(f"❌ 错误: 找不到模型文件 {path}")
        exit()
        
    model = build_model()
    try:
        print(f"🔄 正在加载权重: {os.path.basename(path)}")
        checkpoint = torch.load(path, map_location=device)
        # 兼容处理: 处理带 'state_dict' 键或不带的情况
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"❌ 模型加载失败! 请检查架构是否匹配。\n错误信息: {e}")
        exit()
        
    model.to(device)
    model.eval()
    return model

# =============================================================
# 3. 🖼️ 数据预处理 (512x512)
# =============================================================
# 这里的 Resize 必须是 512，否则模型会报错或效果极差
INFERENCE_TRANSFORM = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def load_mask(mask_path, target_h, target_w):
    try:
        if not os.path.exists(mask_path): return None
        mask = np.load(mask_path)
        if mask.ndim > 2: mask = np.max(mask, axis=-1)
        # 统一缩放到原图大小方便对比
        if mask.shape[:2] != (target_h, target_w):
            mask = cv2.resize(mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.float32)
    except: return None

# =============================================================
# 4. 🔮 预测与可视化核心逻辑
# =============================================================
def predict_and_plot(model, img_path, mask_dir):
    filename = os.path.basename(img_path)
    file_id = os.path.splitext(filename)[0]
    
    # 1. 读取原图
    image = cv2.imread(img_path)
    if image is None: return
    orig_h, orig_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. 尝试读取 Ground Truth 掩码
    # 兼容两种路径: 直接在文件夹下 或 在 forged 子文件夹下
    candidates = [
        os.path.join(mask_dir, file_id + ".npy"),
        os.path.join(mask_dir, "forged", file_id + ".npy")
    ]
    gt_mask = None
    for p in candidates:
        gt_mask = load_mask(p, orig_h, orig_w)
        if gt_mask is not None: break
        
    if gt_mask is None:
        gt_mask = np.zeros((orig_h, orig_w))
        has_gt = False
    else:
        has_gt = True

    # 3. 推理 (Resize -> Predict -> Resize Back)
    aug = INFERENCE_TRANSFORM(image=image_rgb)["image"]
    input_tensor = aug.unsqueeze(0).to(DEVICE) # [1, 3, 512, 512]
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        
    # 获取 512x512 的预测结果
    pred_raw = probs[0][0].cpu().numpy()
    # 还原回原图尺寸 (比如 1024x768)，这样对比才清晰
    pred_mask = cv2.resize(pred_raw, (orig_w, orig_h))
    
    # 4. 绘图
    plt.figure(figsize=(16, 6))
    
    # --- 原图 ---
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title(f"Input: {filename}\n({orig_w}x{orig_h})")
    plt.axis("off")
    
    # --- 真实标签 (GT) ---
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb)
    if has_gt and gt_mask.sum() > 0:
        gt_overlay = np.zeros_like(image_rgb)
        gt_overlay[gt_mask > 0.5] = [0, 255, 0] # 绿色
        plt.imshow(cv2.addWeighted(image_rgb, 0.6, gt_overlay, 0.4, 0))
        plt.title("Ground Truth (Green)")
    else:
        plt.title("Ground Truth (Not Found / Clean)")
    plt.axis("off")
    
    # --- 模型预测 (Pred) ---
    plt.subplot(1, 3, 3)
    plt.imshow(image_rgb)
    
    # 红色热力图显示预测
    pred_overlay = np.zeros_like(image_rgb)
    # 设定一个阈值，比如 0.5
    threshold = 0.5
    mask_binary = pred_mask > threshold
    pred_overlay[mask_binary] = [255, 0, 0] # 红色
    
    plt.imshow(cv2.addWeighted(image_rgb, 0.6, pred_overlay, 0.4, 0))
    
    # 计算 IoU
    iou = 0.0
    if has_gt and gt_mask.sum() > 0:
        intersection = np.logical_and(gt_mask > 0.5, mask_binary).sum()
        union = np.logical_or(gt_mask > 0.5, mask_binary).sum()
        iou = intersection / (union + 1e-6)
        
    plt.title(f"EfficientNet-B4 Prediction\nIoU: {iou:.2f}")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# =============================================================
# 5. 主运行程序
# =============================================================
if __name__ == "__main__":
    # 1. 加载模型
    model = load_checkpoint(MODEL_PATH, DEVICE)
    
    # 2. 获取图片列表
    if not os.path.exists(TEST_IMG_DIR):
        print(f"❌ 图片文件夹不存在: {TEST_IMG_DIR}")
    else:
        # 递归搜索所有图片
        all_files = []
        for root, dirs, files in os.walk(TEST_IMG_DIR):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    all_files.append(os.path.join(root, f))
                    
        print(f"📂 在 {os.path.basename(TEST_IMG_DIR)} 中找到 {len(all_files)} 张图片。")
        
        if len(all_files) > 0:
            # 随机抽取 5 张进行测试
            selected_files = random.sample(all_files, min(5, len(all_files)))
            print("🚀 开始预测...\n")
            
            for img_path in selected_files:
                predict_and_plot(model, img_path, MASK_DIR)
        else:
            print("❌ 文件夹里没有图片")