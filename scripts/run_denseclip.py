"""
Run DenseCLIP inference on a folder of images and save segmentation masks.

Usage:
    python scripts/run_denseclip.py \
        --input  data/pascal_voc/VOCdevkit/VOC2012/JPEGImages/ \
        --prompts "a dog on grass" "a person on a sidewalk" \
        --output results/denseclip/

    # Override default config
    python scripts/run_denseclip.py \
        --config configs/denseclip.yaml \
        --input  data/pascal_voc/VOCdevkit/VOC2012/JPEGImages/ \
        --prompts "a dog on grass" \
        --output results/denseclip/
"""

import argparse
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.denseclip import DenseCLIP


# ── Helpers (same as run_maskclip.py) ────────────────────────────────────────

def save_mask(mask: np.ndarray, output_path: Path):
    """Save a binary mask as a grayscale PNG."""
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img.save(output_path)


def save_overlay(image: Image.Image, mask: np.ndarray, output_path: Path,
                 color=(60, 130, 220), alpha: float = 0.45):
    """
    Save mask overlaid on image as colored PNG.
    DenseCLIP overlays use blue to distinguish from MaskCLIP's red.
    """
    img_array = np.array(image.convert("RGB")).astype(float)
    overlay   = np.zeros_like(img_array)
    for c, v in enumerate(color):
        overlay[:, :, c] = v

    mask_bool = mask.astype(bool)
    result    = img_array.copy()
    result[mask_bool] = (
        img_array[mask_bool] * (1 - alpha) + overlay[mask_bool] * alpha
    )
    Image.fromarray(result.astype(np.uint8)).save(output_path)


def sanitize_prompt(prompt: str) -> str:
    """Turn a prompt string into a safe filename slug."""
    return prompt.lower().replace(" ", "_").replace("/", "-")[:60]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DenseCLIP inference pipeline")
    parser.add_argument("--config",  default="configs/denseclip.yaml")
    parser.add_argument("--input",   required=True, help="Folder of input images")
    parser.add_argument("--prompts", nargs="+",     required=True)
    parser.add_argument("--output",  default="results/denseclip/")
    parser.add_argument("--exts",    nargs="+",     default=[".jpg", ".jpeg", ".png"],
                        help="Image file extensions to include")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device        = config["inference"].get("device", "cpu")
    image_size    = config["inference"].get("image_size", 512)
    save_masks    = config["output"].get("save_masks",    True)
    save_overlays = config["output"].get("save_overlays", True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[DenseCLIP] Loading model on {device}...")
    model = DenseCLIP(
        clip_backbone=config["model"]["clip_backbone"],
        pretrained=config["model"]["pretrained"],
    ).to(device)
    model.eval()
    print(f"[DenseCLIP] Model ready.")

    # Collect image paths
    input_dir   = Path(args.input)
    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in args.exts
    ])

    num_samples = config["dataset"].get("num_samples", -1)
    if num_samples != -1:
        image_paths = image_paths[:num_samples]

    print(f"[DenseCLIP] Found {len(image_paths)} images.")
    print(f"[DenseCLIP] Prompts: {args.prompts}")
    print(f"[DenseCLIP] Output:  {output_dir}\n")

    # Inference loop
    for img_path in tqdm(image_paths, desc="DenseCLIP inference"):
        image = Image.open(img_path).convert("RGB")

        image_resized = image.resize((image_size, image_size), Image.BILINEAR)
        img_tensor    = model.preprocess(image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            masks = model(img_tensor, args.prompts)  # (1, N, H, W) raw logits

        # DenseCLIP outputs raw logits — apply sigmoid before thresholding
        masks = torch.sigmoid(masks)

        for i, prompt in enumerate(args.prompts):
            prompt_slug = sanitize_prompt(prompt)
            mask_raw    = masks[0, i].cpu().numpy()          # (H, W) in [0, 1]
            mask_binary = (mask_raw > 0.5).astype(np.uint8)  # threshold at 0.5

            stem = img_path.stem

            if save_masks:
                mask_dir = output_dir / "masks" / prompt_slug
                mask_dir.mkdir(parents=True, exist_ok=True)
                save_mask(mask_binary, mask_dir / f"{stem}.png")

            if save_overlays:
                overlay_dir = output_dir / "overlays" / prompt_slug
                overlay_dir.mkdir(parents=True, exist_ok=True)
                save_overlay(image_resized, mask_binary,
                             overlay_dir / f"{stem}.png")

    print(f"\n[DenseCLIP] Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
