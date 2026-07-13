"""
Stage 2 Training Script (3D Affordance Alignment)

This stage distills knowledge from the pretrained 2D Branch
into the 3D branch, aligning multi-view 2D representations with 3D features.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
sys.path.append(".")

from utils.utils import seed_torch, read_yaml
from utils.logger import setup_logger
from utils.metrics import evaluating, cal_SIM_3d

from dataset.laso import LasoDataset
from dataset.piad import PiadDataset
from model.branch_2d import Branch2D
from model.branch_3d import Branch3D
from utils.loss import HM_Loss, l1_loss
from utils.align_loss import visibility_weighted_align, cross_view_consistency

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
        train_dataset = PiadDataset(
            cfg["train_split"], cfg["setting"], data_root=cfg["data_root"],
            use_image=use_image, k_images=k_images, img_size=img_size,
        )
        test_dataset = PiadDataset(cfg["test_split"], data_root=cfg["data_root"])
    elif cfg["category"] == "laso":
        train_dataset = LasoDataset(cfg["train_split"], cfg["setting"], data_root=cfg["data_root"])
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


def build_optimizer(model, opt_cfg):
    """Set up optimizer and scheduler for 3D branch."""
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad],
         "lr": opt_cfg["tlr"]},
    ]
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


def train_one_epoch(model_3d, model_2d, loader, optimizer, device, criterion_hm, logger, epoch, train_cfg):
    """
    One training epoch:
      - Freeze 2D branch (teacher)
      - Compute 3D affordance heatmaps and alignment losses
      - NEW: L_Sim  – MIFAG multi-image global cosine invariance
      - NEW: L_aff  – CMAT patch-level affinity matrix alignment
    """
    model_3d.train()
    model_2d.eval()
    loss_sum = 0

    use_image = train_cfg.get("use_image", False)
    k = train_cfg.get("k_images", 1)
    w_sim = train_cfg.get("sim_loss_weight", 0.0)
    w_aff = train_cfg.get("aff_loss_weight", 0.0)

    for _, p in model_2d.named_parameters():
        p.requires_grad = False

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
        pred_3d, feat_3d, gaussian_aff, patch_feat_3d = model_3d(question, point)
        feat_2d, render_feats, alpha, idx, contrib, aff_render, aff_teacher, mask = \
            model_2d(question, point, feat_3d, gaussian_aff)

        # --- Original losses ---
        V = render_feats.shape[0] // B_cur
        C_f, H_f, W_f = render_feats.shape[1], render_feats.shape[2], render_feats.shape[3]

        # A1: visibility/contribution-weighted distillation
        # contrib_a1 = contrib.sum(dim=2).view(-1, 1, H_f, W_f)  # [B*V, 1, H, W]
        # loss_align = visibility_weighted_align(
        #     render_feats, feat_2d, alpha, contrib_a1,
        #     gamma=train_cfg.get("vw_gamma", 1.0))

        # A2: cross-view consistency
        # student_bv = render_feats.view(B_cur, V, C_f, H_f, W_f)
        # loss_cv = cross_view_consistency(student_bv, idx, contrib)

        loss_hm = criterion_hm(pred_3d, label)
        # loss_render = l1_loss(aff_render, aff_teacher.detach(), mask)

        loss = loss_hm
                # + train_cfg["align_loss_weight"]         * loss_align
                # + train_cfg["cv_loss_weight"]            * loss_cv
                # + train_cfg["render_consistency_weight"] * loss_render)

        # --- NEW: L_Sim + L_aff (only when interaction images are available) ---
        loss_sim = torch.tensor(0.0, device=device)
        loss_aff = torch.tensor(0.0, device=device)

        if images is not None and k > 1:
            # Pure affordance text for image branch (strip viewpoint prefix)
            # question[0] is a list of B strings for viewpoint-0; after '.', the affordance part
            pure_queries = [s.split('.')[1].strip() for s in question[0]]
            with torch.no_grad():
                text_img, text_mask_img = model_2d._encode_text(pure_queries, device)  # [B, L, C], [B, L]

                z_imgs = []
                A_2d_list = []
                for j in range(k):
                    img_feat_j = model_2d._encode_image_tokens(images[:, j])     # [B, 256, llm_dim]

                    # L_Sim: global affordance embedding per image
                    z_j, _ = model_2d._image_affordance(text_img, text_mask_img, img_feat_j)  # [B, llm_dim]
                    z_imgs.append(z_j)

                    # L_aff: pool 256 → 128 patches (as in design doc: reshape & mean)
                    patch_2d_j = img_feat_j.reshape(B_cur, 128, 2, -1).mean(2)   # [B, 128, llm_dim]
                    A_2d_list.append(_affinity_matrix(patch_2d_j))               # [B, 128, 128]

            # L_Sim: pairwise cosine distance between global embeddings of the k images
            n_pairs = 0
            for a in range(k):
                for b in range(a + 1, k):
                    loss_sim = loss_sim + (1 - F.cosine_similarity(z_imgs[a], z_imgs[b], dim=-1)).mean()
                    n_pairs += 1
            loss_sim = loss_sim / n_pairs

            # L_aff: MSE between 3D and averaged 2D affinity structures
            A_2d_avg = torch.stack(A_2d_list).mean(0)                           # [B, 128, 128]
            A_3d = _affinity_matrix(patch_feat_3d)                              # [B, 128, 128]
            loss_aff = F.mse_loss(A_3d, A_2d_avg.detach())

            loss = loss + w_sim * loss_sim + w_aff * loss_aff

        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if i % 10 == 0:
            logger.debug(
                f"[Epoch {epoch}] Iter {i}/{len(loader)} | Loss: {loss.item():.4f} "
                f"(hm: {loss_hm.item():.4f}"
                f"sim: {loss_sim.item():.4f}, aff: {loss_aff.item():.4f})"
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
        model_2d.load_state_dict(ckpt["model"], strict=False)

    if train_cfg["resume"]:
        ckpt_path = train_cfg["checkpoint_path"]
        logger.debug(f"Resuming 3D model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model_3d.load_state_dict(ckpt["model"], strict=False)

    optimizer, scheduler = build_optimizer(model_3d, cfg["optimizer"])

    # Count trainable params
    count_trainable_params(model_3d)

    # Training loop
    best_IOU = 0
    save_dir = os.path.join(train_cfg["save_dir"], train_cfg["name"])
    os.makedirs(save_dir, exist_ok=True)

    # Inject dataset image-branch config into train_cfg so train_one_epoch can read them
    train_cfg["use_image"] = cfg["dataset"].get("use_image", False)
    train_cfg["k_images"]  = cfg["dataset"].get("k_images", 1)

    for epoch in range(train_cfg["epochs"]):
        logger.debug(f"Epoch {epoch} start → lr={optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_one_epoch(model_3d, model_2d, train_loader, optimizer, device, criterion_hm, logger, epoch, train_cfg)
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
