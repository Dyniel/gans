# -*- coding: utf-8 -*-
"""
Bloki modelu DC-GAN przyjmujące dowolny rozmiar obrazu
(każda potęga 2 ≥ 32).
"""
import math
import torch.nn as nn


# ---------- pomocnicze bloki ---------- #
def _g_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True),
    )


def _d_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, True),
    )


# ---------- Generator ---------- #
class Generator(nn.Module):
    def __init__(self, img_size: int, latent_dim: int, base_ch: int = 64, img_ch: int = 3):
        """
        img_size – potęga 2 (32-1024); wygenerujemy tyle bloków, by zejść z 4 px do żądanego rozmiaru
        """
        super().__init__()
        assert img_size & (img_size - 1) == 0 and img_size >= 32, "`img_size` musi być potęgą 2 ≥ 32"

        n_ups = int(math.log2(img_size)) - 3  # 4 → 8 → … → img_size
        ch = base_ch * (2 ** n_ups)

        layers = [
            nn.ConvTranspose2d(latent_dim, ch, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
        ]
        for _ in range(n_ups):
            layers.extend(_g_block(ch, ch // 2))
            ch //= 2
        layers.extend(
            [nn.ConvTranspose2d(ch, img_ch, 4, 2, 1, bias=False), nn.Tanh()]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)  # [B,3,H,W]


# ---------- Discriminator ---------- #
class Discriminator(nn.Module):
    def __init__(self, img_size: int, base_ch: int = 64, img_ch: int = 3):
        super().__init__()
        assert img_size & (img_size - 1) == 0 and img_size >= 32

        n_downs = int(math.log2(img_size)) - 3  # img → … → 4
        layers = [
            nn.Conv2d(img_ch, base_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
        ]
        ch = base_ch
        for _ in range(n_downs):
            layers.extend(_d_block(ch, ch * 2))
            ch *= 2
        layers.append(nn.Conv2d(ch, 1, 4, 1, 0, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(x.size(0))  # [B]