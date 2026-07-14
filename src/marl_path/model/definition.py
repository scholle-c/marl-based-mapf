"""Definition of the pathfinding model architecture"""

from __future__ import annotations

import torch
from torch import nn
from abc import ABC, abstractmethod


class DefaultModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DistanceTableCNN(DefaultModel):
    """
    Plain CNN without pooling that maps multi-channel inputs (map, goal,
    start, other agents) to a single-channel [0, 1] delay-probability map.

    Train with forward_logits() + BCEWithLogitsLoss; the resulting
    probability is scaled by an explicit `penalty_scale` hyperparameter
    before being added to h_bfs (see DistTable.compute_delay_model).
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        self._hidden_channels = hidden_channels
        self._depth = depth
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(depth - 1):
            layers.append(
                nn.Conv2d(
                    channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                )
            )
            layers.append(nn.LeakyReLU(inplace=True))
            channels = hidden_channels
        layers.append(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, padding_mode="replicate")
        )
        self.network = nn.Sequential(*layers)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-activation network output — feed this to BCEWithLogitsLoss directly
        (never apply sigmoid before that loss)."""
        return self.network(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, C, H, W) -> (batch, 1, H, W) in [0, 1]."""
        return torch.sigmoid(self.forward_logits(x))


class PatchTransformer(DefaultModel):
    """Small, from-scratch ViT-style model for dense per-cell prediction.

    Not torchvision's vit_b_16/vit_l_32/etc: those are built for 224x224 RGB
    classification (ImageNet-pretrained, single CLS-token output) — neither
    fits here. This is a minimal encoder built for our actual inputs
    (small, non-square, few-channel MAPF grids) and our actual output (one
    value per cell, not one label per image):
      - patch_size=1 (default) treats every grid cell as its own token, so
        there is no upsampling/decoder step needed to get back to per-cell
        resolution (unlike vit_*'s 16px/32px patches, which would collapse a
        32x32 grid to 1-4 tokens total).
      - No pretrained weights: ImageNet statistics have nothing in common
        with binary path/collision masks, so pretraining would not transfer.
      - Output is a linear projection *per token*, reshaped back into
        (1, H, W), not a single global class logit.

    Train with forward_logits() + BCEWithLogitsLoss, same convention as
    DistanceTableCNN.
    """

    def __init__(
        self,
        in_channels: int = 5,
        grid_height: int = 32,
        grid_width: int = 32,
        patch_size: int = 1,
        embed_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if grid_height % patch_size != 0 or grid_width % patch_size != 0:
            raise ValueError(
                f"grid size ({grid_height}x{grid_width}) must be divisible "
                f"by patch_size ({patch_size})"
            )
        self._grid_height = grid_height
        self._grid_width = grid_width
        self._patch_size = patch_size
        self._embed_dim = embed_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._mlp_dim = mlp_dim or embed_dim * 4
        self._n_side_h = grid_height // patch_size
        self._n_side_w = grid_width // patch_size
        n_patches = self._n_side_h * self._n_side_w

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=self._mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, patch_size * patch_size)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        patches = self.patch_embed(x)  # (B, embed_dim, n_side_h, n_side_w)
        tokens = patches.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)  # (B, n_patches, embed_dim)

        out = self.head(tokens)  # (B, n_patches, patch_size*patch_size)
        p = self._patch_size
        out = out.view(batch, self._n_side_h, self._n_side_w, p, p)
        out = out.permute(0, 1, 3, 2, 4).contiguous()
        out = out.view(batch, 1, self._n_side_h * p, self._n_side_w * p)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, C, H, W) -> (batch, 1, H, W) in [0, 1]."""
        return torch.sigmoid(self.forward_logits(x))
