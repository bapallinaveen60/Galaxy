import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Double Conv with Dropout
# -------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),   # 🔥 reduces overfitting
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# -------------------------------
# SE Attention Block
# -------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# -------------------------------
# Improved Attention Fusion
# -------------------------------
class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, 1)
        self.se = SEBlock(channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.se(x)
        s = self.spatial(x)
        return x * s


# -------------------------------
# Upsample Block (Bilinear)
# -------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# -------------------------------
# Dual Stream Attention UNet
# -------------------------------
class DualStreamAttentionUNet(nn.Module):
    def __init__(self, eo_channels=3, sar_channels=1, out_channels=2):
        super().__init__()

        # -------- Encoder --------
        # EO
        self.eo1 = DoubleConv(eo_channels, 32)
        self.eo2 = DoubleConv(32, 64)
        self.eo3 = DoubleConv(64, 128)
        self.eo4 = DoubleConv(128, 256)

        # SAR
        self.sar1 = DoubleConv(sar_channels, 32)
        self.sar2 = DoubleConv(32, 64)
        self.sar3 = DoubleConv(64, 128)
        self.sar4 = DoubleConv(128, 256)

        # Fusion
        self.f1 = AttentionFusion(32)
        self.f2 = AttentionFusion(64)
        self.f3 = AttentionFusion(128)
        self.f4 = AttentionFusion(256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # -------- Decoder --------
        self.up4 = UpBlock(512 + 256, 256)
        self.up3 = UpBlock(256 + 128, 128)
        self.up2 = UpBlock(128 + 64, 64)
        self.up1 = UpBlock(64 + 32, 32)

        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, eo, sar):

        # -------- EO Encoder --------
        e1_eo = self.eo1(eo)
        e2_eo = self.eo2(self.pool(e1_eo))
        e3_eo = self.eo3(self.pool(e2_eo))
        e4_eo = self.eo4(self.pool(e3_eo))

        # -------- SAR Encoder --------
        e1_sar = self.sar1(sar)
        e2_sar = self.sar2(self.pool(e1_sar))
        e3_sar = self.sar3(self.pool(e2_sar))
        e4_sar = self.sar4(self.pool(e3_sar))

        # -------- Fusion --------
        f1 = self.f1(e1_eo, e1_sar)
        f2 = self.f2(e2_eo, e2_sar)
        f3 = self.f3(e3_eo, e3_sar)
        f4 = self.f4(e4_eo, e4_sar)

        # -------- Bottleneck --------
        b = self.bottleneck(self.pool(f4))

        # -------- Decoder --------
        d4 = self.up4(b, f4)
        d3 = self.up3(d4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)

        return self.out(d1)