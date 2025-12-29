import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm 
import warnings

warnings.filterwarnings("ignore")

# 导入之前写好的模块
from DataSet import ForgeryDataset, TRAIN_TRANSFORM
from unet_model import UNet

# ==========================================
# 1. 配置参数
# ==========================================
DATA_ROOT = 'D:/InfSec/Data/recodai-luc-scientific-image-forgery-detection'  
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'train_images')
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, 'train_masks')

BATCH_SIZE = 4       # 批次大小 (显存如果不够可以改小到 2)
LEARNING_RATE = 1e-4 # 学习率
NUM_EPOCHS = 5       # 训练轮数 (先试跑 5 轮看看效果)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在使用设备: {DEVICE}")

def get_all_image_paths(root_dir):
    paths = []
    # 扫描 authentic 和 forged 文件夹
    for sub in ['authentic', 'forged']:
        sub_dir = os.path.join(root_dir, sub)
        if os.path.exists(sub_dir):
            for f in os.listdir(sub_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    paths.append(os.path.join(sub_dir, f))
    return paths

# ==========================================
# 2. 训练循环函数
# ==========================================
def train_fn(loader, model, optimizer, loss_fn, scaler):
    # 优化 tqdm:
    # ncols=100: 固定宽度，防止换行
    # desc="Training": 给进度条左边加个标题
    loop = tqdm(loader, leave=True, ncols=100, desc="Training") 
    total_loss = 0
    
    for batch_idx, (data, targets, _) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # 混合精度训练
        with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")): # 修复警告：改用 torch.amp
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        if DEVICE == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 更新统计
        total_loss += loss.item()
        
        # 优化显示: 只在右边显示 loss，保留 4 位小数
        loop.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(loader)

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == '__main__':
    # --- 准备数据 ---
    all_paths = get_all_image_paths(TRAIN_IMG_DIR)
    print(f"找到 {len(all_paths)} 张图片用于训练。")
    
    if len(all_paths) == 0:
        print("❌ 错误：未找到图片，请检查路径设置！")
        exit()

    # 创建 Dataset 和 DataLoader
    # 为了演示快速开始，暂时用全部数据 (如果太慢，可以 all_paths[:100] 先测试)
    ds = ForgeryDataset(all_paths, TRAIN_MASK_DIR, transform=TRAIN_TRANSFORM)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # --- 初始化模型 ---
    # n_channels=3 (RGB), n_classes=1 (Binary Mask)
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    
    # 定义损失函数 (BCEWithLogitsLoss 结合了 Sigmoid 和 BCE，数值更稳定)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 梯度缩放器 (用于混合精度训练)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    # --- 开始训练 ---
    print("\n🚀 开始训练...")
    for epoch in range(NUM_EPOCHS):
        # 把 Epoch 信息打印在进度条上方，而不是挤在进度条里
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        avg_loss = train_fn(loader, model, optimizer, loss_fn, scaler)
        
        # 打印本轮总结
        print(f"--> Average Loss: {avg_loss:.4f}")
        print(f"-------------------------------------------------------")
        
        # 保存模型
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        # 只保存最新的，或者按 epoch 命名
        torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth.tar")


    print("\n🎉 训练全部完成！")
