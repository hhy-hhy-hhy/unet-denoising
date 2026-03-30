import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model import UNet
from utils import add_gaussian_noise  # 确保 utils.py 中有这个函数

def visualize_single(model, image_path, sigma=25, device='cpu'):
    model.eval()

    # 读取图像
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 等比缩放（长边不超过 512）
    h, w = img.shape[:2]
    max_size = 512
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

    # 归一化到 [0,1]（无论是否缩放，都必须执行）
    img = img.astype(np.float32) / 255.0

    # 添加噪声
    noisy = add_gaussian_noise(img, sigma)

    # 转为 tensor 并推理
    img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device)
    noisy_t = torch.from_numpy(noisy).permute(2,0,1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        denoised_t = model(noisy_t)
    denoised = denoised_t.squeeze(0).permute(1,2,0).cpu().numpy()
    denoised = np.clip(denoised, 0, 1)


    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(noisy)
    axes[1].set_title(f"Noisy (σ={sigma})")
    axes[1].axis('off')
    axes[2].imshow(denoised)
    axes[2].set_title("Denoised")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("single_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("对比图已保存为 single_comparison2.png")

if __name__ == "__main__":
    # 配置路径
    model_path = "D:/U-Net/unet_epoch5.pth"   # 你训练好的模型
    test_img_path = "D:/U-Net/data/test2/monarch.bmp"  # 替换为任意一张测试图片路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    visualize_single(model, Path(test_img_path), sigma=25, device=device)
