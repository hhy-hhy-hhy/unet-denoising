import torch
import numpy as np
import cv2
import time
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from model import UNet
from utils import calculate_psnr, calculate_ssim   # 你的 utils 中已有的函数

# -------------------- 1. 定义测试数据集（返回原图，不裁剪）--------------------
class TestDataset(Dataset):
    """测试数据集：直接返回原图，不做任何裁剪/缩放，也不加噪。"""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))          # 统一缩放到 128×128
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        return torch.from_numpy(img).float()
        img_path = self.image_paths[idx]
        # 读取彩色图
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 归一化到 [0,1]
        img = img.astype(np.float32) / 255.0
        # 转换为 (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float()   
# -------------------- 2. 加载模型 --------------------
def load_model(model_path, device):
    model = UNet(in_channels=3, out_channels=3).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# -------------------- 3. 测试不同噪声强度 --------------------
def test_model(model, test_loader, sigmas=[15, 25, 35, 50], device='cpu'):
    results = []
    with torch.no_grad():
        for sigma in sigmas:
            start_time = time.time()
            psnr_sum = 0.0
            ssim_sum = 0.0
            count = 0

            for clean in test_loader:
                clean = clean.to(device)
                # 添加噪声
                noise = torch.randn_like(clean) * (sigma / 255.0)
                noisy = torch.clamp(clean + noise, 0, 1)

                output = model(noisy)

                # 计算指标（注意：calculate_psnr/ssim 内部会处理好形状）
                psnr_val = calculate_psnr(clean, output, data_range=1.0)
                ssim_val = calculate_ssim(clean, output, data_range=1.0)

                psnr_sum += psnr_val
                ssim_sum += ssim_val
                count += 1

            elapsed = time.time() - start_time
            avg_psnr = psnr_sum / count
            avg_ssim = ssim_sum / count

            results.append({
                'Sigma': sigma,
                'PSNR (dB)': avg_psnr,
                'SSIM': avg_ssim,
                'Time (s)': elapsed
            })
            print(f"Sigma {sigma:2d} | PSNR = {avg_psnr:.2f} dB | SSIM = {avg_ssim:.4f} | Time = {elapsed:.2f} s")

    return results

# -------------------- 4. 主程序 --------------------
if __name__ == "__main__":
    # --- 配置参数（根据你的实际情况修改）---
    model_path = "D:/U-Net/unet_epoch5.pth"      # 训练好的模型路径
    test_dir = Path("D:/U-Net/data/test2")         # 测试图片文件夹
    test_sigmas = [15, 25, 35, 50]                # 要测试的噪声强度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --------------------------------------

    # 获取所有测试图片路径
    test_paths = list(test_dir.glob("*.bmp")) + list(test_dir.glob("*.bmp"))
    print(f"找到 {len(test_paths)} 张测试图片")

    if len(test_paths) == 0:
        print("错误：没有找到测试图片，请检查路径！")
        exit()

    # 创建数据集和数据加载器（batch_size=1，因为图片尺寸可能不同）
    test_dataset = TestDataset(test_paths)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载模型
    print("加载模型...")
    model = load_model(model_path, device)

    # 测试
    print("开始测试...")
    results = test_model(model, test_loader, test_sigmas, device)

    # 保存结果到 CSV
    df = pd.DataFrame(results)
    df.to_csv("test_results_with_time.csv", index=False)
    print("结果已保存至 test_results_with_time.csv")
