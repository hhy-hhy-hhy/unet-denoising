
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import UNet
from dataset import DenoisingDataset
from utils import calculate_psnr, calculate_ssim
from pathlib import Path

def train_model(model, train_loader, epochs=50, lr=1e-4, save_every=5):
    """
    训练模型，并定期保存 checkpoint
    :param model: U-Net 模型
    :param train_loader: 训练数据加载器
    :param epochs: 总训练轮数
    :param lr: 学习率
    :param save_every: 每隔多少个 epoch 保存一次模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Start Training (Fixed Sigma=25)...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # 定期保存模型
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"unet_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # 训练结束保存最终模型
    torch.save(model.state_dict(), "unet_denoise.pth")
    print("Final model saved: unet_denoise.pth")
    return model

def test_generalization(model, test_loader, test_sigmas=[15, 25, 35, 50]):
    """泛化能力测试"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    results = []

    print("Testing Generalization...")
    with torch.no_grad():
        for sigma in test_sigmas:
            psnr_sum, ssim_sum, count = 0, 0, 0

            # 测试集加载器返回干净图像（因为 noise_sigma=0）
            for _, clean_img in test_loader:
                clean_img = clean_img.to(device)
                # 动态添加噪声
                noise = torch.randn_like(clean_img) * (sigma / 255.0)
                noisy_img = torch.clamp(clean_img + noise, 0, 1)

                output = model(noisy_img)

                psnr_sum += calculate_psnr(output, clean_img)
                ssim_sum += calculate_ssim(output, clean_img)
                count += 1

            avg_psnr = psnr_sum / count
            avg_ssim = ssim_sum / count
            results.append({'Sigma': sigma, 'PSNR': avg_psnr.item(), 'SSIM': avg_ssim})
            print(f"Sigma {sigma}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

    return results

if __name__ == "__main__":
    # 数据路径（请根据实际情况修改）
    train_dir = Path("D:/U-Net/data/train")
    test_dir = Path("D:/U-Net/data/test")
    train_paths = list(train_dir.glob("*.png"))
    test_paths = list(test_dir.glob("*.png"))

    print(f"训练集路径: {train_dir}")
    print(f"找到 {len(train_paths)} 张图片")
    # 创建训练数据集（噪声强度固定 25）
    train_dataset = DenoisingDataset(train_paths, noise_sigma=25, patch_size=128)
    print(f"训练数据集样本数: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=3).to(device)

    # 训练（每 5 个 epoch 保存一次）
    trained_model = train_model(model, train_loader, epochs=50, save_every=5)

    # 测试（使用干净图像，测试时动态加噪）
    test_dataset = DenoisingDataset(test_paths, noise_sigma=0, patch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    final_results = test_generalization(trained_model, test_loader)

    # 最终模型已保存在 train_model 中，此处无需重复保存
    print("所有任务完成。")
