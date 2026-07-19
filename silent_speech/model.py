from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

SensorMode = Literal["jaw", "ref", "both"]


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        groups = 8 if channels % 8 == 0 else 4
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x + residual)
        x = self.dropout(x)
        return x * mask


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 32,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        groups = 8 if hidden_channels % 8 == 0 else 4
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    hidden_channels,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in (1, 2, 4, 8)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.stem(x) * mask
        for block in self.blocks:
            x = block(x, mask)
        return x


def masked_mean_max_pool(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # x [B, C, T], mask [B, 1, T]
    denominator = mask.sum(dim=-1).clamp_min(1.0)
    mean = (x * mask).sum(dim=-1) / denominator
    negative_large = torch.finfo(x.dtype).min
    maximum = x.masked_fill(mask == 0, negative_large).amax(dim=-1)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return mean, maximum


class DualBranchTCN(nn.Module):
    """Small paired-IMU TCN with learned cross-sensor interaction.

    For jaw-only or reference-only ablations, only one temporal branch is instantiated.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        sensor_mode: SensorMode = "both",
        hidden_channels: int = 32,
        dropout: float = 0.18,
    ) -> None:
        super().__init__()
        if sensor_mode not in {"jaw", "ref", "both"}:
            raise ValueError(f"Unsupported sensor_mode: {sensor_mode}")
        self.sensor_mode = sensor_mode
        self.hidden_channels = hidden_channels

        if sensor_mode in {"jaw", "both"}:
            self.jaw_encoder = TemporalEncoder(6, hidden_channels, dropout=dropout * 0.67)
        else:
            self.jaw_encoder = None
        if sensor_mode in {"ref", "both"}:
            self.ref_encoder = TemporalEncoder(6, hidden_channels, dropout=dropout * 0.67)
        else:
            self.ref_encoder = None

        if sensor_mode == "both":
            # A learned interaction over jaw, reference, signed difference, and product.
            self.interaction = nn.Sequential(
                nn.Conv1d(hidden_channels * 4, hidden_channels, kernel_size=1, bias=False),
                nn.GroupNorm(8 if hidden_channels % 8 == 0 else 4, hidden_channels),
                nn.SiLU(),
            )
            pooled_channels = hidden_channels * 3 * 2  # 3 streams × mean/max
        else:
            self.interaction = None
            pooled_channels = hidden_channels * 2  # mean/max

        # Add one explicit valid-length fraction after masked pooling.
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_channels + 1),
            nn.Linear(pooled_channels + 1, 96),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # sequence [B, T, 12], mask [B, T]
        if sequence.ndim != 3 or sequence.shape[-1] != 12:
            raise ValueError(f"Expected [B, T, 12], got {tuple(sequence.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"Expected mask [B, T], got {tuple(mask.shape)}")

        mask_c = mask[:, None, :].to(sequence.dtype)
        x = sequence.transpose(1, 2)

        streams: list[torch.Tensor] = []
        if self.sensor_mode in {"jaw", "both"}:
            assert self.jaw_encoder is not None
            jaw = self.jaw_encoder(x[:, :6], mask_c)
            streams.append(jaw)
        if self.sensor_mode in {"ref", "both"}:
            assert self.ref_encoder is not None
            ref = self.ref_encoder(x[:, 6:], mask_c)
            streams.append(ref)

        if self.sensor_mode == "both":
            jaw, ref = streams
            assert self.interaction is not None
            interaction_input = torch.cat([jaw, ref, jaw - ref, jaw * ref], dim=1)
            difference = self.interaction(interaction_input) * mask_c
            streams = [jaw, ref, difference]

        pooled: list[torch.Tensor] = []
        for stream in streams:
            mean, maximum = masked_mean_max_pool(stream, mask_c)
            pooled.extend([mean, maximum])

        length_fraction = mask.to(sequence.dtype).mean(dim=1, keepdim=True)
        fused = torch.cat([*pooled, length_fraction], dim=1)
        return self.classifier(fused)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
