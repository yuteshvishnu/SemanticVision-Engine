"""
Run MaskCLIP inference on a folder of images and save segmentation masks.

Usage:
    python scripts/run_maskclip.py \
        --input  data/pascal_voc/VOCdevkit/VOC2012/JPEGImages/ \
        --prompts "a dog on grass" "a person on a sidewalk" \
        --output results/maskclip/

    # Override default config
    python scripts/run_maskclip.py \
        --config configs/maskclip.yaml \
        --input  data/pascal_voc/VOCdevkit/VOC2012/JPEGImages/ \
        --prompts "a dog on grass" \
        --output results/maskclip/
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

from models.maskclip import MaskCLIP


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_mask(mask: np.ndarray, output_path: Path):
    """Save a binary mask as a grayscale PNG."""
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img.save(output_path)


def save_overlay(image: Image.Image, mask: np.ndarray, output_path: Path,
                 color=(255, 80, 80), alpha: float = 0.45):
    """Save mask overlaid on the original image as a colored PNG."""
    img_array = np.array(image.convert("RGB")).astype(float)
    overlay   = np.zeros_like(img_array)
    for c, v in enumerate(color):
        overlay[:, :, c] = v

    mask_bool = mask.astype(bool)
    result = img_array.copy()
    result[mask_bool] = (
        img_array[mask_bool] * (1 - alpha) + overlay[mask_bool] * alpha
    )
    Image.fromarray(result.astype(np.uint8)).save(output_path)


def sanitize_prompt(prompt: str) -> str:
    """Turn a prompt string into a safe filename slug."""
    return prompt.lower().replace(" ", "_").replace("/", "-")[:60]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MaskCLIP inference pipeline")
    parser.add_argument("--config",   default="configs/maskclip.yaml")
    parser.add_argument("--input",    required=True, help="Folder of input images")
    parser.add_argument("--prompts",  nargs="+",     required=True)
    parser.add_argument("--output",   default="results/maskclip/")
    parser.add_argument("--exts",     nargs="+",     default=[".jpg", ".jpeg", ".png"],
                        help="Image file extensions to include")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device     = config["inference"].get("device", "cpu")
    image_size = config["inference"].get("image_size", 512)
    save_masks    = config["output"].get("save_masks",    True)
    save_overlays = config["output"].get("save_overlays", True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[MaskCLIP] Loading model on {device}...")
    model = MaskCLIP(
        clip_backbone=config["model"]["clip_backbone"],
        pretrained=config["model"]["pretrained"],
    ).to(device)
    model.eval()
    print(f"[MaskCLIP] Model ready.")

    # Collect image paths
    input_dir   = Path(args.input)
    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in args.exts
    ])

    num_samples = config["dataset"].get("num_samples", -1)
    if num_samples != -1:
        image_paths = image_paths[:num_samples]

    print(f"[MaskCLIP] Found {len(image_paths)} images.")
    print(f"[MaskCLIP] Prompts: {args.prompts}")
    print(f"[MaskCLIP] Output:  {output_dir}\n")

    # Inference loop
    for img_path in tqdm(image_paths, desc="MaskCLIP inference"):
        image = Image.open(img_path).convert("RGB")

        # Preprocess: resize to config image_size, apply CLIP transforms
        image_resized = image.resize((image_size, image_size), Image.BILINEAR)
        img_tensor    = model.preprocess(image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            masks = model(img_tensor, args.prompts)  # (1, N, H, W)

        # Save one mask/overlay per prompt
        for i, prompt in enumerate(args.prompts):
            prompt_slug = sanitize_prompt(prompt)
            mask_raw    = masks[0, i].cpu().numpy()         # (H, W) float logits
            mask_binary = (mask_raw > 0).astype(np.uint8)  # threshold at 0

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

    print(f"\n[MaskCLIP] Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
