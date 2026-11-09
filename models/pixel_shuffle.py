# Modified.


import torch
from torch import nn


class PixelShuffle3d(nn.Module):


    def __init__(self, scale):
        super().__init__()
        if not isinstance(scale, int) or scale < 1:
            raise ValueError("scale must be a positive integer")
        self.scale = scale

    def forward(self, input):
        if input.ndim != 5:
            raise ValueError(f"expected NCDHW input, got {tuple(input.shape)}")
        batch, channels, depth, height, width = input.shape
        block = self.scale**3
        if channels % block:
            raise ValueError(f"channels ({channels}) must be divisible by scale^3 ({block})")
        output_channels = channels // block
        return (
            input.reshape(
                batch,
                output_channels,
                self.scale,
                self.scale,
                self.scale,
                depth,
                height,
                width,
            )
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .reshape(
                batch,
                output_channels,
                depth * self.scale,
                height * self.scale,
                width * self.scale,
            )
        )
