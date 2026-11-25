# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import torch.nn.functional as F
from scipy.ndimage import label as ndi_label
from segment_anything import sam_model_registry


class MultiPointSegmenter:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.model.eval()
        self.image_embeddings = None
        self.img_size = None
        self.threshold = threshold

    def set_image(self, image_2d):
        self.img_size = image_2d.shape[:2]
        if image_2d.ndim == 2:
            image_3c = np.repeat(image_2d[:, :, None], 3, axis=-1)
        else:
            image_3c = image_2d
        img_tensor = torch.tensor(image_3c, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.model.device)
        img_resized = F.interpolate(img_tensor, size=(1024, 1024), mode="bilinear", align_corners=False)
        img_min, img_max = torch.min(img_resized), torch.max(img_resized)
        img_norm = (img_resized - img_min) / (img_max - img_min) if img_max > img_min else torch.zeros_like(img_resized)
        with torch.no_grad():
            self.image_embeddings = self.model.image_encoder(img_norm)

    @torch.no_grad()
    def infer_point(self, point):
        x, y = point
        coords = np.array([[[x * 1024 / self.img_size[1], y * 1024 / self.img_size[0]]]], dtype=np.float32)
        coords_torch = torch.tensor(coords, dtype=torch.float32, device=self.model.device)
        labels_torch = torch.tensor([[1]], dtype=torch.long, device=self.model.device)
        sparse, dense = self.model.prompt_encoder(points=(coords_torch, labels_torch), boxes=None, masks=None)
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=self.image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        prob = F.interpolate(torch.sigmoid(low_res_logits), size=self.img_size, mode="bilinear", align_corners=False)
        return prob.detach().cpu().numpy().squeeze()

    def baseline(self, point):
        return (self.infer_point(point) >= self.threshold).astype(np.uint8)

    def average(self, points):
        maps = np.stack([self.infer_point(p) for p in points])
        return (np.mean(maps, axis=0) >= self.threshold).astype(np.uint8)


def define_model(checkpoint_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to(device)
    return model


def random_points(mask, n, rng):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    idx = rng.choice(len(xs), size=n, replace=len(xs) < n)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def dice(pred, gt):
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    return 2.0 * np.sum(pred & gt) / (pred.sum() + gt.sum() + 1e-8)


def save_overlay(img_np, gt_mask, pred_mask, points, out_path, title):
    img = img_np[..., 0] if img_np.ndim == 3 else img_np
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap=ListedColormap(["#ff3b3b"]), alpha=0.40)
    if gt_mask.any():
        ax.contour(gt_mask, levels=[0.5], colors="lime", linewidths=2)
    if len(points):
        pts = np.array(points)
        ax.scatter(pts[:, 0], pts[:, 1], c="yellow", s=40, marker="*", edgecolors="k", linewidths=0.5)
    ax.set_title(title)
    ax.axis("off")
    ax.plot([], [], color="lime", linewidth=2, label="Ground truth (outline)")
    ax.plot([], [], color="#ff3b3b", linewidth=6, label="Prediction (fill)")
    ax.scatter([], [], c="yellow", marker="*", edgecolors="k", label="Point prompts")
    ax.legend(loc="upper right")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def run(args):
    rng = np.random.default_rng(args.seed)
    img_np = np.load(args.image)
    gt_np = np.load(args.gt)
    assert gt_np.ndim == 2, "Ground-truth label map must be 2D"

    labels = [int(v) for v in np.unique(gt_np) if v > 0]
    if not labels:
        raise ValueError("No positive labels found in the ground-truth file.")
    target = args.label if args.label is not None else max(labels, key=lambda l: int(np.sum(gt_np == l)))
    print(f"Labels in GT: {labels} -> segmenting label {target}")
    gt_bin = (gt_np == target).astype(np.uint8)

    model = define_model(args.checkpoint)
    seg = MultiPointSegmenter(model, threshold=args.threshold)
    seg.set_image(img_np)

    pts_path = os.path.join(os.path.dirname(args.image) or ".", "demo_points.npz")
    if os.path.exists(pts_path):
        d = np.load(pts_path)
        baseline_point = tuple(int(v) for v in d["baseline_point"])
        gt_points = [tuple(int(v) for v in p) for p in d["gt_points"]]
    else:
        baseline_point = random_points(gt_bin, 1, rng)[0]
        gt_points = random_points(gt_bin, args.n, rng)

    baseline_mask = seg.baseline(baseline_point)
    baseline_dsc = dice(baseline_mask, gt_bin)

    semi_mask = seg.average(gt_points)
    semi_dsc = dice(semi_mask, gt_bin)

    auto_points = random_points(baseline_mask, args.n, rng)
    auto_mask = seg.average(auto_points) if auto_points else baseline_mask
    auto_dsc = dice(auto_mask, gt_bin)

    print(f"\nlabel {target}")
    print(f"  baseline (1 point)            : DSC = {baseline_dsc:.4f}")
    print(f"  semi-auto GT-guided (N={args.n}) : DSC = {semi_dsc:.4f}   (gain {semi_dsc - baseline_dsc:+.4f})")
    print(f"  automatic (N={args.n})           : DSC = {auto_dsc:.4f}   (gain {auto_dsc - baseline_dsc:+.4f})")

    masks = {"baseline": (baseline_mask, [baseline_point], baseline_dsc),
             "semi": (semi_mask, gt_points, semi_dsc),
             "auto": (auto_mask, auto_points, auto_dsc)}
    chosen_mask, chosen_points, chosen_dsc = masks[args.show]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_overlay(img_np, gt_bin, chosen_mask, chosen_points, args.output,
                 title=f"label {target} | {args.show} N={args.n} | DSC={chosen_dsc:.3f}")
    print(f"\nSaved overlay ({args.show}) -> {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description="Multiple-point MedSAM prompting single-image demo")
    p.add_argument("--image", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--checkpoint", default="checkpoints/medsam_point_prompt_flare22.pth")
    p.add_argument("--label", type=int, default=None)
    p.add_argument("--n", type=int, default=14)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", choices=["baseline", "semi", "auto"], default="semi")
    p.add_argument("--output", default="demo_result.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
