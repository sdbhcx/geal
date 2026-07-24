"""
§9 Plan: Invariant affordance knowledge loss functions.

These losses implement the "Learning 2D Invariant Affordance Knowledge for
3D Affordance Grounding" pipeline (MIFAG-inspired, lightweight variant).

Key design principle: all losses are controlled via config flags.
None of these functions is invoked by default; the training script gates
each call behind the corresponding `use_*_loss` config key.

Reference: https://arxiv.org/abs/2408.13024
"""

import torch
import torch.nn.functional as F


def loss_invariant(z_list, anchor=None, anchor_proj=None, eps=1e-6):
    """
    Invariance constraint: k interaction images of the same affordance
    should map to similar embeddings via the trainable AffordanceProj.

    Includes an anchor term that prevents AffordanceProj from collapsing to
    a constant vector.  ``anchor`` is a list of raw DINO global features
    (same order as z_list); each z_j is pulled toward its own anchor.
    Since anchor dim (768) != z dim (affordance_dim), ``anchor_proj`` projects
    the anchor into the same space before cosine comparison.

    Args:
        z_list: list of [B, C] tensors, one per image (k items total).
        anchor: optional list of [B, C_dino] raw DINO features (same length as z_list).
        anchor_proj: optional nn.Linear(C_dino, C) to project anchor into z space.
                     If anchor is provided but anchor_proj is None, anchor is ignored.

    Returns:
        scalar loss = mean pairwise cosine distance + anchor loss.
    """
    if len(z_list) < 2:
        return torch.tensor(0.0, device=z_list[0].device)

    # Pairwise cosine distance among k image embeddings
    loss = 0.0
    n_pairs = 0
    for a in range(len(z_list)):
        for b in range(a + 1, len(z_list)):
            loss += (1 - F.cosine_similarity(z_list[a], z_list[b], dim=-1)).mean()
            n_pairs += 1
    loss = loss / n_pairs if n_pairs > 0 else torch.tensor(0.0, device=z_list[0].device)

    # Anchor: each z_j should be close to its own raw DINO anchor.
    # Projects anchor into z space first to handle dim mismatch (768 -> affordance_dim).
    if anchor is not None and len(anchor) == len(z_list) and anchor_proj is not None:
        anchor_loss = 0.0
        for z, a in zip(z_list, anchor):
            a_proj = F.normalize(anchor_proj(a.detach()), dim=-1)  # project & normalize
            anchor_loss += (1 - F.cosine_similarity(z, a_proj, dim=-1)).mean()
        anchor_loss = anchor_loss / len(z_list)
        loss = loss + anchor_loss

    return loss


def loss_3d2img(z_img, z_3d, label, img_3d_proj=None, eps=1e-6):
    """
    Core loss: aggregate 3D foreground feature and align it with the
    invariant affordance embedding from interaction images.

    This is the ONLY loss in the pipeline whose gradient reaches `pred_3d`.
    L_invariant shapes z_img quality; L_3d2img propagates that signal to 3D.

    Args:
        z_img: [B, C_img] — invariant affordance embedding (aggregated from k images).
               Will be detached to avoid conflicting gradients with L_invariant.
        z_3d:  [B, N, C_3d] — 3D per-point affordance features (e.g. downsampled_feat
                from Branch3D, where N == num_points in label).
        label: [B, N] — point-level GT (0/1), must match z_3d's point dimension.
        img_3d_proj: optional nn.Linear(C_img, C_3d). When None, C_img must == C_3d.

    Returns:
        scalar mean cosine distance between foreground 3D feature and z_img.
    """
    # Aggregate 3D side foreground features
    fg_mask = (label > 0.5).unsqueeze(-1).float()  # [B, N, 1]
    z_3d_fg = (z_3d * fg_mask).sum(1) / fg_mask.sum(1).clamp_min(eps)  # [B, C_3d]

    z_img_proj = F.normalize(z_img.detach(), dim=-1)
    z_3d_fg_norm = F.normalize(z_3d_fg, dim=-1)

    C_img, C_3d = z_img_proj.shape[-1], z_3d_fg_norm.shape[-1]

    if C_img != C_3d:
        if img_3d_proj is None:
            raise ValueError(
                f"z_img dim ({C_img}) != z_3d dim ({C_3d}). "
                "Pass a Linear(C_img, C_3d) as img_3d_proj."
            )
        z_img_proj = F.normalize(img_3d_proj(z_img_proj), dim=-1)

    loss = (1 - F.cosine_similarity(z_3d_fg_norm, z_img_proj, dim=-1)).mean()
    return loss


def loss_contrastive(z_img, z_3d, label, img_3d_proj=None, temp=0.1, eps=1e-6):
    """
    Contrastive loss using point-level labels to construct positive/negative pairs.

    For each image embedding z_img_j:
      - positive: 3D foreground points (label=1) should be close to z_img_j
      - negative: 3D background points (label=0) should be far from z_img_j

    Uses a per-point cosine similarity against z_img_j with BCE loss.

    Args:
        z_img: [B, k, C_img] — per-image embeddings (one per interaction image).
        z_3d:  [B, N, C_3d] — 3D per-point features.
        label: [B, N] — point-level GT (0/1).
        img_3d_proj: optional nn.Linear(C_img, C_3d). When None, C_img must == C_3d.
        temp:  temperature for cosine similarity.

    Returns:
        scalar contrastive loss.
    """
    B, k, C_img = z_img.shape
    N = z_3d.shape[1]

    z_3d_norm = F.normalize(z_3d, dim=-1)              # [B, N, C_3d]
    z_img_proj = F.normalize(z_img, dim=-1)            # [B, k, C_img]

    C_3d = z_3d_norm.shape[-1]
    if C_img != C_3d:
        if img_3d_proj is None:
            raise ValueError(
                f"z_img dim ({C_img}) != z_3d dim ({C_3d}). "
                "Pass a Linear(C_img, C_3d) as img_3d_proj."
            )
        z_img_proj = F.normalize(img_3d_proj(z_img_proj), dim=-1)  # [B, k, C_3d]

    # Cosine similarity: [B, k, N]
    sim = torch.bmm(z_3d_norm.transpose(1, 2),        # [B, C_3d, N]
                    z_img_proj.transpose(1, 2)).transpose(1, 2)  # [B, k, N]
    sim_scaled = sim / temp

    # Foreground label broadcast across k images: [B, k, N]
    fg_mask = (label > 0.5).unsqueeze(1).expand(-1, k, -1)  # [B, k, N]

    # BCE loss: foreground points should have high similarity
    loss = F.binary_cross_entropy_with_logits(sim_scaled, fg_mask.float())

    return loss
