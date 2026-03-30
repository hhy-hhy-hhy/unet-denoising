import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        # 编码器
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = self._block(128, 256)

        # 解码器
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)   # 拼接后通道数为 128+128=256
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)    # 拼接后通道数为 64+64=128

        # 输出层
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def _block(self, in_ch, out_ch):
        """双层卷积块 + ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码路径
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        # 瓶颈
        b = self.bottleneck(self.pool(e2))

        # 解码路径（含跳跃连接 + 尺寸对齐）
        d2 = self.upconv2(b)
        # 对齐 e2 和 d2 的尺寸
        diffY = e2.size()[2] - d2.size()[2]
        diffX = e2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        # 对齐 e1 和 d1 的尺寸
        diffY = e1.size()[2] - d1.size()[2]
        diffX = e1.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)
