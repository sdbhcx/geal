"""Continuous local 3D tokens with explicit point mappings.

The module keeps local token construction independent from language so it can be
used as the V1 geometry-only branch. Every token records its sampled center and
KNN neighborhood, and can be interpolated back to the original point set.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from model.pointnet2_utils import farthest_point_sample, index_points, square_distance


def validate_local_tokenizer_checkpoint(state_dict: Dict[str, torch.Tensor]) -> None:
    """Fail fast when a V1 evaluation accidentally loads a baseline checkpoint."""
    has_tokenizer = any(key.startswith("local_tokenizer.") for key in state_dict)
    has_fusion = any(key.startswith("token_fusion.") for key in state_dict)
    if not has_tokenizer or not has_fusion:
        raise RuntimeError(
            "local tokenizer evaluation requires checkpoint weights for both "
            "local_tokenizer.* and token_fusion.*"
        )


class Local3DTokenizer(nn.Module):
    """Encode FPS/KNN neighborhoods into continuous local 3D tokens."""

    _SUPPORTED_POOLING = {"max", "mean", "attention"}

    def __init__(
        self,
        token_dim: int,
        num_tokens: int = 256,
        neighbor_k: int = 32,
        hidden_dim: int = 256,
        pooling: str = "max",
        normalize_local_scale: bool = False,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or num_tokens <= 0 or neighbor_k <= 0 or hidden_dim <= 0:
            raise ValueError("token dimensions and neighborhood sizes must be positive")
        if pooling not in self._SUPPORTED_POOLING:
            raise ValueError(
                f"unsupported pooling '{pooling}', expected one of {sorted(self._SUPPORTED_POOLING)}"
            )

        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.neighbor_k = neighbor_k
        self.pooling = pooling
        self.normalize_local_scale = normalize_local_scale

        self.local_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
            nn.GELU(),
        )
        self.token_norm = nn.LayerNorm(token_dim)
        self.attention_score = nn.Linear(token_dim, 1) if pooling == "attention" else None

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build local tokens.

        Args:
            xyz: Point coordinates shaped ``[B, N, 3]``.

        Returns:
            token_features: Continuous features shaped ``[B, C, M]``.
            metadata: Explicit center, neighborhood and relative-coordinate maps.
        """
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must have shape [B, N, 3], got {tuple(xyz.shape)}")
        if xyz.shape[1] == 0:
            raise ValueError("xyz must contain at least one point")

        point_count = xyz.shape[1]
        token_count = min(self.num_tokens, point_count)
        neighbor_count = min(self.neighbor_k, point_count)

        center_idx = farthest_point_sample(xyz, token_count)
        centers = index_points(xyz, center_idx)
        distance = square_distance(centers, xyz).clamp_min_(0.0)
        neighbor_dist, neighbor_idx = distance.topk(
            k=neighbor_count, dim=-1, largest=False, sorted=True
        )
        grouped_xyz = index_points(xyz, neighbor_idx)
        relative_xyz = grouped_xyz - centers.unsqueeze(2)

        if self.normalize_local_scale:
            local_scale = relative_xyz.norm(dim=-1).amax(dim=2, keepdim=True).clamp_min(1e-6)
            encoded_relative_xyz = relative_xyz / local_scale.unsqueeze(-1)
        else:
            local_scale = torch.ones(
                (*relative_xyz.shape[:2], 1), dtype=xyz.dtype, device=xyz.device
            )
            encoded_relative_xyz = relative_xyz

        local_features = self.local_encoder(encoded_relative_xyz)
        if self.pooling == "max":
            token_features = local_features.amax(dim=2)
        elif self.pooling == "mean":
            token_features = local_features.mean(dim=2)
        else:
            attention = torch.softmax(self.attention_score(local_features), dim=2)
            token_features = torch.sum(attention * local_features, dim=2)

        token_features = self.token_norm(token_features).transpose(1, 2).contiguous()
        metadata = {
            "center_idx": center_idx,
            "centers": centers,
            "neighbor_idx": neighbor_idx,
            "neighbor_dist": neighbor_dist,
            "relative_xyz": relative_xyz,
            "local_scale": local_scale,
        }
        return token_features, metadata


class TokenToPointInterpolator(nn.Module):
    """Interpolate local token features to all original points."""

    def __init__(self, interpolate_k: int = 3, eps: float = 1e-8) -> None:
        super().__init__()
        if interpolate_k <= 0:
            raise ValueError("interpolate_k must be positive")
        self.interpolate_k = interpolate_k
        self.eps = eps

    def forward(
        self,
        xyz: torch.Tensor,
        centers: torch.Tensor,
        token_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Interpolate ``[B, C, M]`` token features to ``[B, C, N]`` points."""
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must have shape [B, N, 3], got {tuple(xyz.shape)}")
        if centers.ndim != 3 or centers.shape[-1] != 3:
            raise ValueError(
                f"centers must have shape [B, M, 3], got {tuple(centers.shape)}"
            )
        if token_features.ndim != 3:
            raise ValueError(
                f"token_features must have shape [B, C, M], got {tuple(token_features.shape)}"
            )
        if token_features.shape[0] != xyz.shape[0] or centers.shape[0] != xyz.shape[0]:
            raise ValueError("xyz, centers and token_features must share the batch dimension")
        if token_features.shape[-1] != centers.shape[1]:
            raise ValueError("token_features token count must match centers")
        if centers.shape[1] == 0:
            raise ValueError("centers must contain at least one token")

        interpolation_count = min(self.interpolate_k, centers.shape[1])
        distance = square_distance(xyz, centers).clamp_min_(0.0)
        point_dist, point_idx = distance.topk(
            k=interpolation_count, dim=-1, largest=False, sorted=True
        )

        inverse_distance = 1.0 / (point_dist + self.eps)
        weight = inverse_distance / inverse_distance.sum(dim=-1, keepdim=True)
        token_features_bmc = token_features.transpose(1, 2)
        gathered_features = index_points(token_features_bmc, point_idx)
        point_features = torch.sum(gathered_features * weight.unsqueeze(-1), dim=2)
        point_features = point_features.transpose(1, 2).contiguous()

        metadata = {
            "point_to_token_idx": point_idx,
            "point_to_token_dist": point_dist,
            "point_to_token_weight": weight,
        }
        return point_features, metadata


class TokenFusion(nn.Module):
    """Fuse token-derived dense features with the PointNet++ residual stream."""

    _SUPPORTED_MODES = {"add", "gated_residual"}

    def __init__(
        self,
        channels: int,
        mode: str = "gated_residual",
        init_value: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"unsupported fusion mode '{mode}', expected one of {sorted(self._SUPPORTED_MODES)}"
            )

        self.mode = mode
        self.token_projection = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(init_value)))
        if mode == "gated_residual":
            self.gate = nn.Conv1d(channels * 2, channels, kernel_size=1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        else:
            self.gate = None

    def forward(
        self, pointnet_features: torch.Tensor, token_point_features: torch.Tensor
    ) -> torch.Tensor:
        if pointnet_features.shape != token_point_features.shape:
            raise ValueError(
                "pointnet_features and token_point_features must have the same shape, "
                f"got {tuple(pointnet_features.shape)} and {tuple(token_point_features.shape)}"
            )

        token_residual = self.token_projection(token_point_features)
        if self.gate is not None:
            gate = torch.sigmoid(
                self.gate(torch.cat([pointnet_features, token_residual], dim=1))
            )
            token_residual = gate * token_residual
        return pointnet_features + self.residual_scale * token_residual
