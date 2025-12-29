import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from DataSet import ForgeryDataset, TRAIN_TRANSFORM, visualize_sample
from Rle import rle_encode, rle_decode

DATA_ROOT = 'D:/InfSec/Data/recodai-luc-scientific-image-forgery-detection' 
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'train_images')
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, 'train_masks')


# ----------------------------------------------------
# 3. EDA 可视化
# ----------------------------------------------------
def visualize_sample(image, mask):
    """
    显示图像和叠加了伪造区域的掩码。
    """
    # 确保 NumPy 格式: H x W x C (RGB)
    if isinstance(image, torch.Tensor):
        # 如果是 PyTorch Tensor (C, H, W)，转回 NumPy (H, W, C)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8) # 去归一化
    
    plt.figure(figsize=(10, 5))
    
    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    # 叠加掩码
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    
    # 创建红色半透明叠加层
    mask_colored = np.zeros_like(image, dtype=np.uint8)
    mask_colored[mask > 0] = [255, 0, 0] # 红色
    
    # 使用 cv2.addWeighted 叠加
    blended = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
    
    plt.imshow(blended)
    plt.title("Image with Forgery Mask (Red)")
    plt.axis('off')
    plt.show()

# ----------------------------------------------------
# 4. 执行流程 (Main Execution Block)
# ----------------------------------------------------
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # 确保导入了 RLE 函数 
    try:
        from Rle import rle_encode, rle_decode
    except ImportError:
        print("提示: 未找到 rle_utils 模块，跳过 RLE 验证步骤。")
        rle_encode = None

    # ==========================================
    # 第一步：扫描所有子文件夹收集图片路径
    # ==========================================
    authentic_dir = os.path.join(TRAIN_IMG_DIR, 'authentic')
    forged_dir = os.path.join(TRAIN_IMG_DIR, 'forged')
    
    print(f"正在准备扫描图片根目录: {TRAIN_IMG_DIR}")
    all_image_paths = []

    # 扫描 authentic (真实) 文件夹
    if os.path.exists(authentic_dir):
        files = os.listdir(authentic_dir)
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                all_image_paths.append(os.path.join(authentic_dir, f))
    
    # 扫描 forged (伪造) 文件夹
    if os.path.exists(forged_dir):
        files = os.listdir(forged_dir)
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                all_image_paths.append(os.path.join(forged_dir, f))

    # 检查是否找到图片
    total_imgs = len(all_image_paths)
    print(f"总共找到 {total_imgs} 张图片路径.")
    if total_imgs == 0:
        print("❌ 错误：未找到图片，请检查路径！")
        exit()

    # ==========================================
    # 第二步：初始化数据集
    # ==========================================
    # 为了演示，取前 20 张 

    print("\n正在初始化 Dataset...")
    train_dataset = ForgeryDataset(
        image_paths=test_paths,  
        mask_dir=TRAIN_MASK_DIR,
        transform=TRAIN_TRANSFORM 
    )

    # ==========================================
    # 第三步：EDA 与 RLE 编码器验证
    # ==========================================
    print("\n--- 探索性数据分析 (EDA) 与 RLE 验证 ---")
    
    # 尝试找几个包含“伪造”内容的样本来测试 RLE
    found_forgery_sample = False
    
    # 遍历前 5 个样本 (或者直到找到一个伪造样本)
    for i in range(min(5, len(train_dataset))):
        try:
            image_tensor, mask_tensor, img_id = train_dataset[i]
            
            # 检查是否有伪造区域 (max > 0)
            has_forgery = mask_tensor.max() > 0
            label_str = "⚠️ 伪造 (Forgery)" if has_forgery else "✅ 真实 (Authentic)"
            
            print(f"[{i}] Case ID: {img_id} | 类型: {label_str}")

            # ---------------------------
            # RLE 验证逻辑 (恢复的部分)
            # ---------------------------
            if has_forgery and rle_encode is not None:
                found_forgery_sample = True
                print("    -> 检测到伪造区域，正在测试 RLE 编码...")
                
                # 1. 准备数据: 强制转为 0/1 的 uint8
                mask_np = (mask_tensor[0].numpy() > 0.5).astype(np.uint8)
                
                # 2. 编码
                rle_str = rle_encode(mask_np)
                print(f"    -> RLE 字符串示例: {rle_str[:50]} ...")
                
                # 3. 解码验证
                decoded_mask = rle_decode(rle_str, mask_np.shape)
                
                # 4. 对比 (确保两个都是 uint8)
                if np.array_equal(mask_np, decoded_mask):
                    print("    -> ✅ RLE 编码/解码验证成功！")
                else:
                    print("    -> ❌ 警告：RLE 解码不一致！")
                    # 调试：打印一下差异
                    diff = np.sum(mask_np != decoded_mask)
                    print(f"       差异像素数: {diff}")
        except Exception as e:
            print(f"❌ 处理样本 {i} 时出错: {e}")

    if not found_forgery_sample:
        print("提示: 前几个样本中全是真实图片，因此跳过了 RLE 测试。这很正常。")

    # ==========================================
    # 第四步：构建 DataLoader (用于实际训练)
    # ==========================================
    print("\n--- 构建 DataLoader (实际训练管道测试) ---")
    try:
        # 定义 DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=4,      # 批次大小
            shuffle=True,      # 打乱数据
            num_workers=0      # Windows下建议先设为0，避免多进程报错；Linux可设为4
        )
        
        print(f"DataLoader 构建成功。批次大小: {train_loader.batch_size}")
        print("正在尝试读取一个批次的数据...")
        
        # 尝试从 Loader 中获取一个 Batch
        images, masks, ids = next(iter(train_loader))
        
        print(f"✅ 成功加载一个批次!")
        print(f"    图像 Tensor 形状 (Batch, C, H, W): {images.shape}")
        print(f"    掩码 Tensor 形状 (Batch, 1, H, W): {masks.shape}")
        print(f"    本批次的 Case IDs: {ids}")
        
    except Exception as e:
        print(f"❌ DataLoader 测试失败: {e}")
        import traceback

        traceback.print_exc()
