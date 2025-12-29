import numpy as np

# 替换文件路径
data = np.load('D:/InfSec/Data/recodai-luc-scientific-image-forgery-detection/train_masks/90.npy')

# 现在 'data' 就是一个 NumPy 数组，你可以像操作普通数组一样操作它
print(data.shape)  # 查看数组的形状/维度
print(data.dtype)  # 查看数组的数据类型

print(data)    # 查看前5个元素
