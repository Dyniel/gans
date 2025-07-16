# -*- coding: utf-8 -*-
"""
Bloki modelu GANformer.
"""
import torch
import torch.nn as nn

# Prosty blok transformera
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs):
        attn_output, _ = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

# Generator GANformera
class Generator(nn.Module):
    def __init__(self, latent_dim=128, img_size=16, embed_dim=256, num_heads=4, ff_dim=1024, num_blocks=4, out_channels=3):
        super(Generator, self).__init__()
        self.entry = nn.Linear(latent_dim, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_blocks)])
        self.out = nn.Linear(embed_dim, img_size * img_size * out_channels)
        self.out_channels = out_channels
        self.img_size = img_size

    def forward(self, z):
        z = z.squeeze() # usunięcie wymiarów 1x1
        x = self.entry(z)
        x = x.unsqueeze(0) # dodanie wymiaru sekwencji
        for block in self.blocks:
            x = block(x)
        x = x.squeeze(0) # usunięcie wymiaru sekwencji
        x = self.out(x)
        return x.view(-1, self.out_channels, self.img_size, self.img_size)

# Dyskryminator GANformera
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, img_size=16, embed_dim=256, num_heads=4, ff_dim=1024, num_blocks=4):
        super(Discriminator, self).__init__()
        self.entry = nn.Linear(img_size * img_size * in_channels, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_blocks)])
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.entry(x)
        x = x.unsqueeze(0) # dodanie wymiaru sekwencji
        for block in self.blocks:
            x = block(x)
        x = x.squeeze(0) # usunięcie wymiaru sekwencji
        return self.out(x).squeeze()
