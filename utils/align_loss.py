"""
Direction-A alignment losses:
  A1 — visibility/contribution-weighted distillation (replaces global MSE)
  A2 — cross-view consistency regularisation (activates map_back_feat)
"""

import torch
from renderer.render_utils import map_back_feat


def visibility_weighted_align(student, teacher, alpha, contrib, gamma=1.0, eps=1e-6):
    """A1: Visibility/contribution-weighted MSE distillation.

    Only pixels with high foreground opacity (alpha) and reliable rendering
    (contrib) contribute strongly to the loss; background and occluded thin
    regions are down-weighted automatically.

    Args:
        student:  [Bn, C, H, W]  3D features projected to 2D (student)
        teacher:  [Bn, C, H, W]  DINO cross-modal features (teacher)
        alpha:    [Bn, 1, H, W]  per-pixel cumulative opacity from the rasterizer
        contrib:  [Bn, 1, H, W]  per-pixel total contribution weight (sum over n_contrib)
        gamma:    float           exponent on contrib (0 → pure foreground mask, 1 → default)
        eps:      float           numerical stability

    Returns:
        Scalar loss.
    """
    # Detach weights: only feature differences drive gradients
    w = alpha.detach() * contrib.detach().clamp_min(0).pow(gamma)  # [Bn, 1, H, W]
    se = ((student - teacher) ** 2).sum(1, keepdim=True)           # [Bn, 1, H, W]
    return (w * se).sum() / (w.sum() + eps)


def cross_view_consistency(student, idx, contrib, num_points=2048, min_views=2, eps=1e-6):
    """A2: Cross-view consistency via per-point feature variance minimisation.

    Backprojects each view's student features to 3D using rendered_idx /
    rendered_contrib, then penalises the variance of the same 3D point's
    representation across views.  Only points visible in at least `min_views`
    views contribute to the loss.

    Args:
        student: [B, V, C, H, W]  per-view student feature maps
        idx:     [B, V, n_contrib, H, W]  point indices (rendered_idx)
        contrib: [B, V, n_contrib, H, W]  contribution weights (rendered_contrib)
        num_points: int  number of 3D points (N)
        min_views:  int  minimum views a point must appear in to be counted
        eps:        float

    Returns:
        Scalar loss.
    """
    # Backproject each view to 3D: feats_v [B, V, N, C], vis_v [B, V, N]
    feats_v, vis_v = map_back_feat(student, idx, contrib, num_points)

    # Binary visibility mask: point visible in view v if total contrib > 0
    m = (vis_v > 0).float()                                        # [B, V, N]

    # Weighted mean feature across views
    denom = m.sum(1).clamp_min(eps)                                # [B, N]
    fbar = (feats_v * m.unsqueeze(-1)).sum(1) / denom.unsqueeze(-1)  # [B, N, C]

    # Per-view squared deviation from mean, masked to visible views
    var = ((feats_v - fbar.unsqueeze(1)) ** 2).sum(-1) * m        # [B, V, N]

    # Average variance per point (over visible views)
    per_point = var.sum(1) / denom                                 # [B, N]

    # Only penalise points seen in at least min_views views
    valid = (m.sum(1) >= min_views).float()                        # [B, N]
    return (per_point * valid).sum() / (valid.sum() + eps)
