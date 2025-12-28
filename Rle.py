import numpy as np

def rle_encode(mask):
    """
    输入: binary mask (0 or 1), shape (H, W)
    输出: RLE string
    """
    # 强制将掩码转换为 0/1 的整数，防止传入 0/255 或 float
    pixels = mask.flatten(order='F') # 使用 Fortran order (列优先) 展平
    pixels = (pixels > 0.5).astype(np.int8) # 阈值化，确保是二值
    
    # 在开头和结尾填充 0，方便找变化点
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    
    # RLE 格式: start length start length ...
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

def rle_decode(rle_str, shape):
    """
    输入: RLE string, shape (H, W)
    输出: binary mask (0 or 1)
    """
    if not isinstance(rle_str, str) or rle_str == 'authentic':
        return np.zeros(shape, dtype=np.uint8)
    
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
        
    return mask.reshape(shape, order='F')