"""
Generate side-by-side visual comparisons of MaskCLIP vs DenseCLIP masks.
Output is a 3-panel figure: Original | MaskCLIP (red) | DenseCLIP (blue)

These figures go directly into the midterm and final report.

Usage:
    # Single image
    python scripts/compare_outputs.py \
        --image   data/pascal_voc/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg \
        --prompt  "a dog sitting on grass" \
        --output  results/comparison/

    # Batch — all images in a folder
    python scripts/compare_outputs.py \
        --input   data/pascal_voc/VOCdevkit/VOC2012/JPEGImages/ \
        --prompt  "a dog sitting on grass" \
        --output  results/comparison/ \
        --limit   20
"""

import argparse
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.maskclip import MaskCLIP
from models.denseclip import DenseCLIP
from evaluation.metrics import compute_iou


# ── Overlay helper ────────────────────────────────────────────────────────────

def apply_overlay(image: np.ndarray, mask: np.ndarray,
                  color: tuple, alpha: float = 0.45) -> np.ndarray:
    """Blend a colored mask over an RGB image array."""
    result  = image.astype(float).copy()
    overlay = np.zeros_like(image, dtype=float)
    for c, v in enumerate(color):
        overlay[:, :, c] = v

    mask_bool = mask.astype(bool)
    result[mask_bool] = (
        result[mask_bool] * (1 - alpha) + overlay[mask_bool] * alpha
    )
    return result.astype(np.uint8)


# ── Single comparison figure ──────────────────────────────────────────────────

def plot_comparison(
    image: Image.Image,
    prompt: str,
    mc_mask: np.ndarray,
    dc_mask: np.ndarray,
    output_path: Path,
    mc_iou: float = None,
    dc_iou: float = None,
    mc_clip_sim: float = None,
    dc_clip_sim: float = None,
):
    """
    Save a 3-panel comparison figure.

    Panel 1: Original image
    Panel 2: MaskCLIP mask overlay (red)
    Panel 3: DenseCLIP mask overlay (blue)

    Args:
        image:       PIL Image (original)
        prompt:      text prompt used for segmentation
        mc_mask:     (H, W) binary MaskCLIP mask
        dc_mask:     (H, W) binary DenseCLIP mask
        output_path: path to save the PNG
        mc_iou:      optional MaskCLIP IoU to show in subtitle
        dc_iou:      optional DenseCLIP IoU to show in subtitle
        mc_clip_sim: optional MaskCLIP CLIP similarity
        dc_clip_sim: optional DenseCLIP CLIP similarity
    """
    img_array = np.array(image.convert("RGB"))

    mc_overlay = apply_overlay(img_array, mc_mask, color=(255, 80, 80))
    dc_overlay = apply_overlay(img_array, dc_mask, color=(60, 130, 220))

    fig = plt.figure(figsize=(15, 5), facecolor="white")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.03)

    # Panel 1 — original
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_array)
    ax1.set_title("Original", fontsize=13, fontweight="bold", pad=8)
    ax1.axis("off")

    # Panel 2 — MaskCLIP
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(mc_overlay)
    mc_title = "MaskCLIP"
    mc_stats = []
    if mc_iou      is not None: mc_stats.append(f"IoU={mc_iou:.3f}")
    if mc_clip_sim is not None: mc_stats.append(f"CLIP={mc_clip_sim:.3f}")
    if mc_stats:
        mc_title += f"\n{' · '.join(mc_stats)}"
    ax2.set_title(mc_title, fontsize=12, fontweight="bold",
                  color="#993C1D", pad=8, linespacing=1.4)
    ax2.axis("off")

    # Panel 3 — DenseCLIP
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(dc_overlay)
    dc_title = "DenseCLIP"
    dc_stats = []
    if dc_iou      is not None: dc_stats.append(f"IoU={dc_iou:.3f}")
    if dc_clip_sim is not None: dc_stats.append(f"CLIP={dc_clip_sim:.3f}")
    if dc_stats:
        dc_title += f"\n{' · '.join(dc_stats)}"
    ax3.set_title(dc_title, fontsize=12, fontweight="bold",
                  color="#0C447C", pad=8, linespacing=1.4)
    ax3.axis("off")

    fig.suptitle(f'Prompt: "{prompt}"', fontsize=11,
                 style="italic", color="#555", y=1.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Side-by-side mask comparison")
    # Single image mode
    parser.add_argument("--image",   default=None, help="Path to a single image")
    # Batch mode
    parser.add_argument("--input",   default=None, help="Folder of images (batch mode)")
    parser.add_argument("--limit",   type=int, default=20,
                        help="Max images to compare in batch mode")
    # Shared
    parser.add_argument("--prompt",  required=True, help="Text prompt for segmentation")
    parser.add_argument("--output",  default="results/comparison/")
    parser.add_argument("--mc_config", default="configs/maskclip.yaml")
    parser.add_argument("--dc_config", default="configs/denseclip.yaml")
    args = parser.parse_args()

    # Load configs
    with open(args.mc_config) as f: mc_cfg = yaml.safe_load(f)
    with open(args.dc_config) as f: dc_cfg = yaml.safe_load(f)

    device     = mc_cfg["inference"].get("device", "cpu")
    image_size = mc_cfg["inference"].get("image_size", 512)

    # Load both models
    print("Loading MaskCLIP...")
    mc_model = MaskCLIP(
        clip_backbone=mc_cfg["model"]["clip_backbone"],
        pretrained=mc_cfg["model"]["pretrained"],
    ).to(device)
    mc_model.eval()

    print("Loading DenseCLIP...")
    dc_model = DenseCLIP(
        clip_backbone=dc_cfg["model"]["clip_backbone"],
        pretrained=dc_cfg["model"]["pretrained"],
    ).to(device)
    dc_model.eval()

    # Collect images
    if args.image:
        image_paths = [Path(args.image)]
    elif args.input:
        image_paths = sorted(Path(args.input).glob("*.jpg"))[:args.limit]
        image_paths += sorted(Path(args.input).glob("*.png"))[:args.limit]
        image_paths = image_paths[:args.limit]
    else:
        parser.error("Provide either --image or --input")

    output_dir = Path(args.output)
    print(f"\nComparing {len(image_paths)} image(s) for prompt: \"{args.prompt}\"\n")

    for img_path in tqdm(image_paths, desc="Comparing"):
        image         = Image.open(img_path).convert("RGB")
        image_resized = image.resize((image_size, image_size), Image.BILINEAR)
        img_tensor    = mc_model.preprocess(image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            mc_masks = mc_model(img_tensor, [args.prompt])       # (1,1,H,W) cosine sim
            dc_masks = torch.sigmoid(
                dc_model(img_tensor, [args.prompt])              # (1,1,H,W) logits → probs
            )

        mc_mask = (mc_masks[0, 0].cpu().numpy() > 0.0).astype(np.uint8)
        dc_mask = (dc_masks[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

        out_path = output_dir / f"{img_path.stem}.png"
        plot_comparison(
            image=image_resized,
            prompt=args.prompt,
            mc_mask=mc_mask,
            dc_mask=dc_mask,
            output_path=out_path,
        )

    print(f"\nComparisons saved to: {output_dir}")
    print(f"Tip: use these figures directly in your midterm report.")


if __name__ == "__main__":
    main()
