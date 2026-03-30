import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2, data_range=1.0):
    """
    计算两张图像之间的 PSNR（峰值信噪比）
    参数:
        img1, img2: torch.Tensor 或 numpy.ndarray，形状可以是 (C, H, W) 或 (H, W, C)
        data_range: 图像像素值范围，如果图像归一化到 [0,1] 则取 1.0，若 [0,255] 则取 255.0
    返回:
        psnr_value: float
    """
    # 如果输入是 tensor，转为 numpy
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()
    # 确保形状为 (H, W, C)（skimage 要求）
    if img1.ndim == 3 and img1.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
    elif img1.ndim == 4 and img1.shape[1] == 3:  # (B, C, H, W) -> 取第一张并转
        img1 = img1[0].transpose(1, 2, 0)
        img2 = img2[0].transpose(1, 2, 0)
    # 计算 PSNR
    return psnr(img1, img2, data_range=data_range)

def calculate_ssim(img1, img2, data_range=1.0):
    """
    计算两张图像之间的 SSIM（结构相似性）
    参数:
        img1, img2: torch.Tensor 或 numpy.ndarray，形状 (C, H, W) 或 (H, W, C)
        data_range: 图像像素值范围，归一化到 [0,1] 时取 1.0，[0,255] 时取 255.0
    返回:
        ssim_value: float
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()
    # 确保形状为 (H, W, C)
    if img1.ndim == 3 and img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
    elif img1.ndim == 4 and img1.shape[1] == 3:
        img1 = img1[0].transpose(1, 2, 0)
        img2 = img2[0].transpose(1, 2, 0)
    # 多通道 SSIM，设置 channel_axis=-1 表示最后一个维度是通道
    return ssim(img1, img2, data_range=data_range, channel_axis=-1)

def add_gaussian_noise(image, sigma, clip=True):
    """
    给图像添加高斯噪声（支持 numpy 或 tensor）
    参数:
        image: numpy.ndarray 或 torch.Tensor，形状 (H,W,C) 或 (C,H,W)
        sigma: 噪声标准差（像素值范围 0-255）
        clip: 是否将结果截断到 [0,1]
    返回:
        noisy: 与 image 同类型同形状
    """
    if torch.is_tensor(image):
        noise = torch.randn_like(image) * (sigma / 255.0)
        noisy = image + noise
        if clip:
            noisy = torch.clamp(noisy, 0, 1)
    else:
        noise = np.random.normal(0, sigma/255.0, image.shape)
        noisy = image + noise
        if clip:
            noisy = np.clip(noisy, 0, 1)
    return noisy
