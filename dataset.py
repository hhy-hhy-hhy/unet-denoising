import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class DenoisingDataset(Dataset):
    def __init__(self, image_paths, noise_sigma=25, patch_size=128, transform=None):
        """
        Args:
            image_paths: 图片路径列表
            noise_sigma: 训练时固定噪声强度，测试时可设为 0（不加噪）或外部动态加噪
            patch_size: 固定裁剪尺寸 (宽高相同)
            transform: 可选的数据增强
        """
        self.image_paths = image_paths
        self.noise_sigma = noise_sigma
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        # 每个 epoch 每张图采样 10 个 patch（可根据需要调整）
        return len(self.image_paths) * 10

    def __getitem__(self, idx):
        # 循环使用图片
        img_path = self.image_paths[idx % len(self.image_paths)]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0   # 归一化到 [0,1]

        h, w, c = img.shape

        # 随机裁剪或缩放至固定大小
        if h >= self.patch_size and w >= self.patch_size:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            img = img[top:top+self.patch_size, left:left+self.patch_size, :]
        else:
            # 如果原图小于 patch_size，则缩放（较少见）
            img = cv2.resize(img, (self.patch_size, self.patch_size))

        # 转为 (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        # 添加噪声（仅当 noise_sigma > 0 时）
        if self.noise_sigma > 0:
            noise = np.random.normal(0, self.noise_sigma / 255.0, img.shape)
            noisy = img + noise
            noisy = np.clip(noisy, 0, 1)
        else:
            noisy = img  # 测试时可能不加噪，但外部会动态加噪

        return torch.from_numpy(noisy).float(), torch.from_numpy(img).float()
