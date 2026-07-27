"""
Stage 2 Training Script (3D Affordance Alignment)

This stage distills knowledge from the pretrained 2D Branch
into the 3D branch, aligning multi-view 2D representations with 3D features.
"""

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
sys.path.append(".")

from utils.utils import seed_torch, read_yaml
from utils.logger import setup_logger
from utils.metrics import evaluating, cal_SIM_3d
from utils.clip_text_encoder import remap_text_encoder_keys

from dataset.laso import LasoDataset
from dataset.piad import PiadDataset
from model.branch_2d import Branch2D
from model.branch_3d import Branch3D
from utils.loss import HM_Loss, l1_loss
from utils.align_loss import visibility_weighted_align, cross_view_consistency
from utils.affordance_loss import loss_invariant, loss_3d2img, loss_contrastive
from model.iam import IAM
from model.adm import ADM

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def count_trainable_params(model):
    """Print number of trainable parameters per submodule."""
    for name, module in model.named_children():
        if any(p.requires_grad for p in module.parameters()):
            num = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"Module: {name:20s} | Trainable params: {num / 1e6:.2f}M")


def build_dataloader(cfg):
    """Initialize train/test dataloaders."""
    if cfg["category"] == "piad":
        use_image = cfg.get("use_image", False)
        k_images  = cfg.get("k_images", 1)
        img_size  = cfg.get("img_size", 224)
        use_augmented = cfg.get("use_augmented", False)
        n_aug_q = cfg.get("n_augmented_questions", 50)
        use_func_desc = cfg.get("use_functional_desc", False)
        func_desc_strat = cfg.get("func_desc_strategy", "prefix")
        train_dataset = PiadDataset(
            cfg["train_split"], cfg["setting"], data_root=cfg["data_root"],
            use_image=use_image, k_images=k_images, img_size=img_size,
            use_augmented=use_augmented, n_augmented_questions=n_aug_q,
            use_functional_desc=use_func_desc, func_desc_strategy=func_desc_strat,
        )
        test_dataset = PiadDataset(cfg["test_split"], data_root=cfg["data_root"])
    elif cfg["category"] == "laso":
        use_augmented = cfg.get("use_augmented", False)
        n_aug_q = cfg.get("n_augmented_questions", 50)
        use_func_desc = cfg.get("use_functional_desc", False)
        func_desc_strat = cfg.get("func_desc_strategy", "prefix")
        train_dataset = LasoDataset(
            cfg["train_split"], cfg["setting"], data_root=cfg["data_root"],
            use_augmented=use_augmented, n_augmented_questions=n_aug_q,
            use_functional_desc=use_func_desc, func_desc_strategy=func_desc_strat,
        )
        test_dataset = LasoDataset(cfg["test_split"], data_root=cfg["data_root"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
    )
    return train_loader, test_loader


def build_optimizer(model_3d, opt_cfg, model_2d=None):
    """Set up optimizer and scheduler.

    model_2d: optional Branch2D whose trainable params (e.g. AffordanceProj)
    should also be included in the optimizer.
    """
    param_dicts = [
        {"params": [p for n, p in model_3d.named_parameters()
                    if "text_encoder" not in n and p.requires_grad]},
        {"params": [p for n, p in model_3d.named_parameters()
                    if "text_encoder" in n and p.requires_grad],
         "lr": opt_cfg["tlr"]},
    ]
    if model_2d is not None:
        param_dicts.append({
            "params": [p for p in model_2d.parameters() if p.requires_grad],
            "lr": opt_cfg["lr"],
        })

    optimizer = torch.optim.Adam(
        params=param_dicts,
        lr=opt_cfg["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=opt_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opt_cfg["step_size"], gamma=opt_cfg["gamma"]
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------
# Training and Evaluation Loops
# ---------------------------------------------------------------------

def _affinity_matrix(feat):
    """
    Compute patch-level cosine affinity matrix.
    Args:
        feat: [B, m, C]  (already projected to a common dim)
    Returns:
        A:    [B, m, m]  values in [-1, 1]
    """
    f = F.normalize(feat, dim=-1)
    return torch.bmm(f, f.transpose(1, 2))


def train_one_epoch(model_3d, model_2d, loader, optimizer, device, criterion_hm, logger, epoch, train_cfg,
                    img_3d_proj=None, anchor_proj=None,
                    iam=None, adm=None, iam_proj=None, point_proj=None,
                    iam_loss_weight=0.5, loss_3d2img_weight=0.5):
    """
    One training epoch:
      - Freeze 2D branch (teacher)
      - Compute 3D affordance heatmaps and alignment losses

    Loss pipeline is fully controlled by config flags:
      - use_new_losses: master switch for the §9 invariant-affordance pipeline
        (replaces the legacy L_sim/L_aff when True)
      - use_invariant_loss / use_3d2img_loss / use_contrastive_loss: individual
        loss toggles (only active when use_new_losses=True)
      - All weight parameters read from config; defaults to 0 (no-op).
    """
    model_3d.train()
    model_2d.eval()
    loss_sum = 0

    use_image = train_cfg.get("use_image", False)
    k = train_cfg.get("k_images", 1)

    # --- §9 Pipeline flags (config-controlled) ---
    use_iam_adm    = train_cfg.get("use_iam_adm", False)
    use_new_losses = train_cfg.get("use_new_losses", False) and not use_iam_adm
    use_invariant  = train_cfg.get("use_invariant_loss", False) and use_new_losses
    use_3d2img     = train_cfg.get("use_3d2img_loss", False) and use_new_losses
    use_contrast   = train_cfg.get("use_contrastive_loss", False) and use_new_losses

    w_inv  = train_cfg.get("inv_loss_weight", 0.1)
    w_3d   = train_cfg.get("loss_3d2img_weight", 0.5)
    w_con  = train_cfg.get("contrastive_loss_weight", 0.1)

    # --- Legacy L_sim/L_aff (only when new pipeline is disabled) ---
    w_sim = train_cfg.get("sim_loss_weight", 0.0) if not use_new_losses else 0.0
    w_aff = train_cfg.get("aff_loss_weight", 0.0) if not use_new_losses else 0.0

    # Freeze entire 2D branch, then selectively unfreeze AffordanceProj
    for _, p in model_2d.named_parameters():
        p.requires_grad = False
    if use_new_losses and hasattr(model_2d, "affordance_proj"):
        for p in model_2d.affordance_proj.parameters():
            p.requires_grad = True

    for i, batch in enumerate(loader):
        if use_image and k > 1:
            point, _, _, question, _, label, images = batch
            images = images.to(device)   # [B, k, 3, H, W]
        else:
            point, _, _, question, _, label = batch[:6]
            images = None

        optimizer.zero_grad()
        point, label = point.to(device), label.to(device)
        B_cur = point.shape[0]

        # --- Forward ---
        # feat_3d = downsampled_feat [B, 2048, 64] — full point-set, matches label
        # fused_feat = Branch3D internal fused features [B, 512, 2048] — for IAM/ADM
        pred_3d, feat_3d, gaussian_aff, patch_feat_3d, fused_feat = model_3d(question, point)
        feat_2d, render_feats, alpha, idx, contrib, aff_render, aff_teacher, mask = \
            model_2d(question, point, feat_3d, gaussian_aff)

        # --- Original heatmap loss ---
        loss_kld = nn.MSELoss()(render_feats, feat_2d)
        loss_hm = criterion_hm(pred_3d, label)
        loss = loss_hm + train_cfg["kl_loss_weight"]*loss_kld

        # =========================================================================
        # §9 Pipeline: IAM+ADM (primary) OR lightweight L_invariant + L_3d2img
        # =========================================================================
        loss_iam  = torch.tensor(0.0, device=device)
        loss_3d_val = torch.tensor(0.0, device=device)

        if images is not None and k > 1 and use_iam_adm and iam is not None:
            # --- IAM+ADM pipeline ---
            pure_queries = [s.split('.')[1].strip() for s in question[0]]
            with torch.no_grad():
                text_img, text_mask_img = model_2d._encode_text(pure_queries, device)

            # Extract DINO features and project to IAM dim
            dino_feats = []
            for j in range(k):
                with torch.no_grad():
                    dino_feat_j = model_2d.get_raw_dino_features(images[:, j])  # [B, P, 768]
                dino_feats.append(iam_proj(dino_feat_j))  # [B, P, iam.dim]

            # IAM: extract invariant affordance knowledge
            iam_out = iam(dino_feats)
            image_queries = iam_out["image_queries"]      # [k, B, P, iam.dim]
            loss_iam = iam_out["similarity_loss"]

            # ADM: fuse IAM queries with 3D point features
            # fused_feat: [B, 512, 2048] → 降采样点数 → transpose → Linear(512, adm.dim) → transpose back
            adm_max_pts = train_cfg.get("adm", {}).get("max_points", None)
            if adm_max_pts is not None:
                num_pts = fused_feat.shape[-1]
                if num_pts > adm_max_pts:
                    idx = torch.randint(0, num_pts, (adm_max_pts,), device=device)
                    fused_feat = fused_feat[:, :, idx]
            point_tokens = point_proj(fused_feat.transpose(1, 2)).transpose(1, 2)  # [B, iam.dim, N]
            P_enhanced = adm(image_queries, point_tokens)  # [B, iam.dim, N]

            # L_3d2img: align enhanced 3D features with label
            # P_enhanced: [B, iam.dim, N] → transpose → [B, N, iam.dim] matches z_3d shape
            if loss_3d2img_weight > 0:
                loss_3d_val = loss_3d2img(
                    iam_out["invariant_features"],  # z_img: [B, iam.dim]
                    P_enhanced.transpose(1, 2),      # z_3d: [B, N, iam.dim]
                    label
                )

            loss = (loss
                    + iam_loss_weight * loss_iam
                    + loss_3d2img_weight * loss_3d_val)

        elif images is not None and k > 1 and use_new_losses and iam is None:
            # --- Lightweight pipeline (fallback) ---
            pure_queries = [s.split('.')[1].strip() for s in question[0]]
            with torch.no_grad():
                text_img, text_mask_img = model_2d._encode_text(pure_queries, device)

            z_imgs = []
            dino_anchors = []
            for j in range(k):
                with torch.no_grad():
                    dino_feat_j = model_2d.get_raw_dino_features(images[:, j])
                global_feat = dino_feat_j.mean(1)  # [B, 768]
                dino_anchors.append(F.normalize(global_feat, dim=-1))
                z_j = F.normalize(model_2d.affordance_proj(global_feat), dim=-1)
                z_imgs.append(z_j)

            z_img_mean = torch.stack(z_imgs).mean(0)
            if use_invariant:
                loss_inv = loss_invariant(z_imgs, anchor=dino_anchors, anchor_proj=anchor_proj)
                loss = loss + w_inv  * loss_inv
            if use_3d2img:
                loss_3d_val = loss_3d2img(z_img_mean, feat_3d, label, img_3d_proj=img_3d_proj)
                loss = loss + w_3d   * loss_3d_val
            if use_contrast:
                z_imgs_stacked = torch.stack(z_imgs)
                loss_con_val = loss_contrastive(z_imgs_stacked, feat_3d, label,
                                               img_3d_proj=img_3d_proj)
                loss = loss + w_con  * loss_con_val
            # loss = (loss
            #         + w_inv  * loss_inv
            #         + w_3d   * loss_3d_val
            #         + w_con  * loss_con_val)

        # =========================================================================
        # Legacy pipeline: L_sim + L_aff (only when use_new_losses=False)
        # =========================================================================
        loss_sim = torch.tensor(0.0, device=device)
        loss_aff = torch.tensor(0.0, device=device)

        if images is not None and k > 1 and not use_new_losses and (w_sim > 0 or w_aff > 0):
            pure_queries = [s.split('.')[1].strip() for s in question[0]]
            with torch.no_grad():
                text_img, text_mask_img = model_2d._encode_text(pure_queries, device)

                z_imgs_legacy = []
                A_2d_list = []
                for j in range(k):
                    img_feat_j = model_2d._encode_image_tokens(images[:, j])

                    z_j, _ = model_2d._image_affordance(text_img, text_mask_img, img_feat_j)
                    z_imgs_legacy.append(z_j)

                    patch_2d_j = img_feat_j.reshape(B_cur, 128, 2, -1).mean(2)
                    A_2d_list.append(_affinity_matrix(patch_2d_j))

            n_pairs = 0
            for a in range(k):
                for b in range(a + 1, k):
                    loss_sim = loss_sim + (1 - F.cosine_similarity(z_imgs_legacy[a], z_imgs_legacy[b], dim=-1)).mean()
                    n_pairs += 1
            loss_sim = loss_sim / n_pairs

            A_2d_avg = torch.stack(A_2d_list).mean(0)
            A_3d = _affinity_matrix(patch_feat_3d)
            loss_aff = F.mse_loss(A_3d, A_2d_avg.detach())

            loss = loss + w_sim * loss_sim + w_aff * loss_aff

        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if i % 10 == 0:
            if use_iam_adm and iam is not None:
                logger.debug(
                    f"[Epoch {epoch}] Iter {i}/{len(loader)} | Loss: {loss.item():.4f} "
                    f"(hm: {loss_hm.item():.4f}, iam: {loss_iam.item():.4f}, "
                    f"3d2img: {loss_3d_val.item():.4f})"
                )
            elif use_new_losses and iam is None:
                logger.debug(
                    f"[Epoch {epoch}] Iter {i}/{len(loader)} | Loss: {loss.item():.4f} "
                    f"(hm: {loss_hm.item():.4f}, inv: {loss_inv.item():.4f}, "
                    f"3d2img: {loss_3d_val.item():.4f}, con: {loss_con_val.item():.4f})"
                )
            else:
                logger.debug(
                    f"[Epoch {epoch}] Iter {i}/{len(loader)} | Loss: {loss.item():.4f} "
                    f"(hm: {loss_hm.item():.4f}, sim: {loss_sim.item():.4f}, aff: {loss_aff.item():.4f})"
                )

    return loss_sum / len(loader)


def evaluate(model_3d, loader, device, criterion_hm, logger):
    """
    Validation loop:
      - Computes IOU, SIM, MAE, and AUC across all test samples.
    """
    model_3d.eval()
    results, targets = [], []
    total_mae, total_points = 0, 0

    with torch.no_grad():
        for i, (point, _, _, question, _, label) in enumerate(loader):

            point, label = point.to(device), label.to(device)
            pred = model_3d(question, point)

            val_loss = criterion_hm(pred, label)
            mae, n_pts = evaluating(pred, label)
            total_mae += mae.item()
            total_points += n_pts

            # 按样本展开，避免不同 batch 形状不一致导致 np.array 失败
            results.extend(list(pred.cpu().numpy()))
            targets.extend(list(label.cpu().numpy()))

            logger.debug(f"[Val] Batch {i}/{len(loader)} | Loss: {val_loss.item():.4f}")

    mean_mae = total_mae / total_points

    # Compute similarity and AUC/IOU
    sim_scores = np.array([cal_SIM_3d(r, t) for r, t in zip(results, targets)])
    SIM = np.nanmean(sim_scores)

    IOUs, AUCs = [], []
    IOU_thres = np.linspace(0, 1, 20)

    for t_true, p_score in zip(targets, results):
        t_true = (t_true >= 0.5).astype(int)     # 逐样本二值化
        if np.sum(t_true) == 0:
            continue
        auc = roc_auc_score(t_true.flatten(), p_score.flatten())
        AUCs.append(auc)
        temp_iou = []
        for thr in IOU_thres:
            p_mask = (p_score >= thr).astype(int)
            intersect = np.sum(p_mask & t_true)
            union = np.sum(p_mask | t_true)
            temp_iou.append(intersect / (union + 1e-6))
        IOUs.append(np.mean(temp_iou))

    IOU = np.nanmean(IOUs)
    AUC = np.nanmean(AUCs)
    logger.debug(f"Validation → IOU: {IOU:.4f}, AUC: {AUC:.4f}, SIM: {SIM:.4f}, MAE: {mean_mae:.4f}")
    return IOU, mean_mae


# ---------------------------------------------------------------------
# Main Training Entry
# ---------------------------------------------------------------------

def main(cfg_path="config/train_stage2.yaml"):
    """
    Stage 2: 3D Affordance Alignment Training
    """
    cfg = read_yaml(cfg_path)
    train_cfg = cfg["train"]

    # Select device
    gpu_id = str(train_cfg.get("gpu", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"[INFO] Using GPU {gpu_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_torch(train_cfg["seed"])
    logger, sign = setup_logger(train_cfg)

    train_loader, test_loader = build_dataloader({**cfg["dataset"], "batch_size": train_cfg["batch_size"]})

    # Build models
    model_2d = Branch2D(cfg["model_2d"], cfg["renderer"]).to(device)
    model_3d = Branch3D(cfg["model_3d"]).to(device)
    criterion_hm = HM_Loss().to(device)

    # Load pretrained 2D weights (frozen teacher)
    if train_cfg.get("pretrained_2d", None):
        ckpt_path = train_cfg["pretrained_2d"]
        logger.debug(f"Loading pretrained 2D model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model_2d.load_state_dict(remap_text_encoder_keys(ckpt["model"]), strict=False)

    if train_cfg["resume"]:
        ckpt_path = train_cfg["checkpoint_path"]
        logger.debug(f"Resuming 3D model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model_3d.load_state_dict(remap_text_encoder_keys(ckpt["model"]), strict=False)

    # Build optimizer: include model_2d trainable params (AffordanceProj)
    # only when the §9 pipeline is enabled.
    use_new_losses = train_cfg.get("use_new_losses", False)
    model_2d_for_opt = model_2d if use_new_losses else None
    optimizer, scheduler = build_optimizer(model_3d, cfg["optimizer"], model_2d_for_opt)

    # §9 Pipeline: img_3d_proj (AffordanceProj → project_dim) created once,
    # persists across epochs, so it actually learns.
    img_3d_proj = None
    if use_new_losses and train_cfg.get("use_3d2img_loss", False):
        if hasattr(model_2d, "affordance_proj"):
            img_3d_proj = torch.nn.Linear(model_2d.affordance_dim,
                                          cfg["model_3d"].get("project_dim", 64)).to(device)
            torch.nn.init.kaiming_normal_(img_3d_proj.weight)
            img_3d_proj.train()
            optimizer.add_param_group({"params": img_3d_proj.parameters(),
                                       "lr": cfg["optimizer"]["lr"]})

    # §9 Pipeline: anchor_proj (dino_dim → affordance_dim) for L_invariant
    # projects raw DINO anchors into the affordance space. Created once,
    # persists across epochs, so it actually learns.
    anchor_proj = None
    if use_new_losses and train_cfg.get("use_invariant_loss", False):
        if hasattr(model_2d, "affordance_proj"):
            anchor_proj = torch.nn.Linear(cfg["model_2d"].get("dino_dim", 768),
                                          model_2d.affordance_dim).to(device)
            torch.nn.init.kaiming_normal_(anchor_proj.weight)
            anchor_proj.train()
            optimizer.add_param_group({"params": anchor_proj.parameters(),
                                       "lr": cfg["optimizer"]["lr"]})

    # IAM+ADM Pipeline (replaces lightweight §9 when use_iam_adm=True)
    use_iam_adm = train_cfg.get("use_iam_adm", False)
    iam_cfg = cfg.get("iam", {})
    adm_cfg = cfg.get("adm", {})
    k = cfg["dataset"].get("k_images", 3)

    iam = None
    adm = None
    iam_proj = None
    point_proj = None
    iam_loss_weight = train_cfg.get("iam_loss_weight", 0.5)
    loss_3d2img_weight = train_cfg.get("loss_3d2img_weight", 0.5)

    if use_iam_adm and iam_cfg.get("use", False) and adm_cfg.get("use", False):
        iam_dim = iam_cfg.get("dim", 768)
        adm_dim = adm_cfg.get("dim", iam_dim)

        # IAM: extract invariant affordance knowledge from DINO patch features
        iam = IAM(
            dim=iam_dim,
            num_heads=iam_cfg.get("num_heads", 6),
            qkv_bias=iam_cfg.get("qkv_bias", False),
            invariant_extract_layers=iam_cfg.get("invariant_extract_layers", 5),
            image_count=k,
            image_token_count=iam_cfg.get("image_token_count", 16),
            drop_rate=iam_cfg.get("drop_rate", 0.0),
        ).to(device)
        iam.train()

        # ADM: fuse IAM queries with 3D point features
        adm = ADM(
            dim=adm_dim,
            attention_drop=adm_cfg.get("attention_drop", 0.1),
        ).to(device)
        adm.train()

        # Projection layers (config-controlled dims)
        # iam_proj: DINO output dim (768) → IAM dim
        dino_out_dim = cfg["model_2d"].get("dino_dim", 768)
        iam_proj = torch.nn.Linear(dino_out_dim, iam_dim).to(device)
        torch.nn.init.kaiming_normal_(iam_proj.weight)
        iam_proj.train()

        # point_proj: Branch3D fused_feat dim (512) → ADM dim
        branch3d_dim = cfg["model_3d"].get("emb_dim", 512)
        point_proj = torch.nn.Linear(branch3d_dim, adm_dim).to(device)
        torch.nn.init.kaiming_normal_(point_proj.weight)
        point_proj.train()

        # Add IAM+ADM params to optimizer
        optimizer.add_param_group({"params": iam.parameters(),
                                    "lr": cfg["optimizer"]["lr"]})
        optimizer.add_param_group({"params": adm.parameters(),
                                    "lr": cfg["optimizer"]["lr"]})
        optimizer.add_param_group({"params": iam_proj.parameters(),
                                    "lr": cfg["optimizer"]["lr"]})
        optimizer.add_param_group({"params": point_proj.parameters(),
                                    "lr": cfg["optimizer"]["lr"]})

        logger.debug(f"IAM+ADM built: iam.dim={iam_dim}, adm.dim={adm_dim}, iam_proj={dino_out_dim}→{iam_dim}, point_proj={branch3d_dim}→{adm_dim}")

    # Count trainable params
    count_trainable_params(model_3d)
    if use_new_losses:
        count_trainable_params(model_2d)
    if use_iam_adm:
        count_trainable_params(iam)
        count_trainable_params(adm)

    # Training loop
    best_IOU = 0
    save_dir = os.path.join(train_cfg["save_dir"], train_cfg["name"])
    os.makedirs(save_dir, exist_ok=True)

    # Inject dataset image-branch config into train_cfg so train_one_epoch can read them
    train_cfg["use_image"] = cfg["dataset"].get("use_image", False)
    train_cfg["k_images"]  = cfg["dataset"].get("k_images", 1)

    for epoch in range(train_cfg["epochs"]):
        logger.debug(f"Epoch {epoch} start → lr={optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_one_epoch(model_3d, model_2d, train_loader, optimizer, device, criterion_hm, logger, epoch, train_cfg,
                                     img_3d_proj=img_3d_proj, anchor_proj=anchor_proj,
                                     iam=iam, adm=adm, iam_proj=iam_proj, point_proj=point_proj,
                                     iam_loss_weight=iam_loss_weight, loss_3d2img_weight=loss_3d2img_weight)
        IOU, mae = evaluate(model_3d, test_loader, device, criterion_hm, logger)
        scheduler.step()

        if IOU > best_IOU:
            best_IOU = IOU
            model_path = os.path.join(save_dir, f"best_model_{sign}.pt")
            torch.save({
                "model": model_3d.state_dict(),
                "optimizer": optimizer.state_dict(),
                "Epoch": epoch
            }, model_path)
            logger.debug(f"New best model saved → IOU={best_IOU:.4f} | {model_path}")

    logger.debug(f"Training complete. Best IOU: {best_IOU:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_stage2.yaml")
    args = parser.parse_args()
    main(args.config)
