"""
Main evaluation runner — compares MaskCLIP and DenseCLIP across all metrics.

Usage:
    # Run single model
    python evaluation/evaluate.py --model maskclip

    # Run both models
    python evaluation/evaluate.py --model all

    # Run with prompt robustness variants
    python evaluation/evaluate.py --model all --prompts configs/prompt_variants.yaml
"""

import argparse
import json
import sys
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    compute_iou,
    compute_miou,
    compute_robustness_variance,
    CLIPSimilarityScorer,
)


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_pascal_voc_val(voc_root: str, num_samples: int = 200):
    """
    Load PASCAL VOC 2012 validation image paths and ground truth mask paths.

    Args:
        voc_root:    path to VOCdevkit/VOC2012/
        num_samples: how many samples to evaluate (-1 = full val set)

    Returns:
        list of dicts: [{"image_path": ..., "mask_path": ..., "image_id": ...}]
    """
    voc_root  = Path(voc_root)
    val_file  = voc_root / "ImageSets" / "Segmentation" / "val.txt"
    img_dir   = voc_root / "JPEGImages"
    mask_dir  = voc_root / "SegmentationClass"

    with open(val_file) as f:
        image_ids = [line.strip() for line in f.readlines()]

    if num_samples != -1:
        image_ids = image_ids[:num_samples]

    samples = []
    for img_id in image_ids:
        img_path  = img_dir  / f"{img_id}.jpg"
        mask_path = mask_dir / f"{img_id}.png"
        if img_path.exists() and mask_path.exists():
            samples.append({
                "image_id":   img_id,
                "image_path": str(img_path),
                "mask_path":  str(mask_path),
            })

    return samples


def load_gt_mask(mask_path: str, class_idx: int) -> np.ndarray:
    """
    Load a PASCAL VOC ground truth segmentation mask for a specific class.

    PASCAL VOC masks are palette PNGs where pixel value = class index.
    255 = ignore (boundary pixels).

    Args:
        mask_path:  path to SegmentationClass PNG
        class_idx:  VOC class index (1=aeroplane, 12=dog, 15=person, etc.)

    Returns:
        binary mask: (H, W) numpy array {0, 1}
    """
    mask = np.array(Image.open(mask_path))
    binary = (mask == class_idx).astype(np.uint8)
    return binary


# VOC class name → index mapping (1-indexed, 0 = background)
VOC_CLASSES = {
    "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5,
    "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10,
    "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15,
    "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20,
}


# ── Single model evaluation ───────────────────────────────────────────────────

def run_evaluation(model_name: str, config: dict, prompts: dict) -> dict:
    """
    Full evaluation pipeline for a single model.

    Args:
        model_name: "maskclip" or "denseclip"
        config:     loaded YAML config dict
        prompts:    dict of {class_name: {"base": ..., "paraphrase": ..., ...}}

    Returns:
        structured results dict with per-image and aggregate metrics
    """
    import torch

    device = config["inference"].get("device", "cpu")

    # Load model
    if model_name == "maskclip":
        from models.maskclip import MaskCLIP
        model = MaskCLIP(
            clip_backbone=config["model"]["clip_backbone"],
            pretrained=config["model"]["pretrained"],
        ).to(device)
    else:
        from models.denseclip import DenseCLIP
        model = DenseCLIP(
            clip_backbone=config["model"]["clip_backbone"],
            pretrained=config["model"]["pretrained"],
        ).to(device)

    model.eval()

    # Load dataset
    samples = load_pascal_voc_val(
        voc_root=config["dataset"]["root"],
        num_samples=config["dataset"]["num_samples"],
    )

    clip_scorer = CLIPSimilarityScorer(device=device)

    results       = []
    all_ious      = []
    all_clip_sims = []
    robustness_per_class = {}

    for sample in tqdm(samples, desc=f"[{model_name}]"):
        image = Image.open(sample["image_path"]).convert("RGB")
        img_tensor = model.preprocess(image).unsqueeze(0).to(device)

        for class_name, prompt_variants in prompts.items():
            gt_mask    = load_gt_mask(sample["mask_path"], VOC_CLASSES[class_name])

            # Skip images that don't contain this class
            if gt_mask.sum() == 0:
                continue

            variant_ious = {}

            for variant_name, prompt_text in prompt_variants.items():
                with torch.no_grad():
                    masks = model(img_tensor, [prompt_text])  # (1, 1, H, W)

                pred_mask = masks[0, 0].cpu().numpy()

                # Resize gt_mask to match pred if needed
                H, W = pred_mask.shape
                from PIL import Image as PILImage
                gt_resized = np.array(
                    PILImage.fromarray(gt_mask).resize((W, H), PILImage.NEAREST)
                )

                iou = compute_iou(pred_mask, gt_resized)
                variant_ious[variant_name] = iou

            # Base prompt CLIP similarity on region crop
            base_prompt  = prompt_variants["base"]
            base_mask_bin = (
                torch.sigmoid(
                    torch.tensor(
                        np.array(Image.fromarray(
                            (list(variant_ious.values())[0] > 0).astype(np.uint8) * 255
                        ))
                    )
                ).numpy() > 0.5
            ).astype(np.uint8)

            # Crop region for CLIP similarity
            from explanation.crop_regions import crop_from_mask
            crop = crop_from_mask(image, base_mask_bin)
            clip_sim = clip_scorer.score(crop, base_prompt) if crop else 0.0

            robustness = compute_robustness_variance(variant_ious)

            result_entry = {
                "image_id":   sample["image_id"],
                "class":      class_name,
                "iou_scores": variant_ious,
                "base_iou":   variant_ious.get("base", 0.0),
                "clip_sim":   round(clip_sim, 4),
                "robustness": robustness,
            }
            results.append(result_entry)

            all_ious.append(variant_ious.get("base", 0.0))
            all_clip_sims.append(clip_sim)

            if class_name not in robustness_per_class:
                robustness_per_class[class_name] = []
            robustness_per_class[class_name].append(robustness["variance"])

    aggregate = {
        "miou":                    round(float(np.mean(all_ious)), 4) if all_ious else 0.0,
        "mean_clip_sim":           round(float(np.mean(all_clip_sims)), 4) if all_clip_sims else 0.0,
        "mean_robustness_variance": round(float(np.mean([
            v for vals in robustness_per_class.values() for v in vals
        ])), 4) if robustness_per_class else 0.0,
        "per_class_robustness":    {
            cls: round(float(np.mean(vals)), 4)
            for cls, vals in robustness_per_class.items()
        },
    }

    print(f"\n[{model_name}] mIoU: {aggregate['miou']} | "
          f"CLIP sim: {aggregate['mean_clip_sim']} | "
          f"Robustness var: {aggregate['mean_robustness_variance']}")

    return {
        "model":     model_name,
        "per_image": results,
        "aggregate": aggregate,
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results: dict):
    """Print a clean results table to stdout."""
    print("\n" + "=" * 65)
    print(f"{'Model':<15} {'mIoU':>8} {'CLIP Sim':>10} {'Robustness Var':>16}")
    print("=" * 65)
    for model_name, res in all_results.items():
        agg = res.get("aggregate", {})
        print(
            f"{model_name:<15}"
            f"{agg.get('miou', 'N/A'):>8}"
            f"{agg.get('mean_clip_sim', 'N/A'):>10}"
            f"{agg.get('mean_robustness_variance', 'N/A'):>16}"
        )
    print("=" * 65 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate MaskCLIP and DenseCLIP")
    parser.add_argument("--model",   choices=["maskclip", "denseclip", "all"], default="all")
    parser.add_argument("--config",  type=str, default=None)
    parser.add_argument("--prompts", type=str, default="configs/prompt_variants.yaml")
    parser.add_argument("--output",  type=str, default="results/full_eval.json")
    args = parser.parse_args()

    # Load prompt variants
    with open(args.prompts) as f:
        prompt_config = yaml.safe_load(f)
    prompts = prompt_config["prompts"]

    models = ["maskclip", "denseclip"] if args.model == "all" else [args.model]
    all_results = {}

    for model_name in models:
        config_path = args.config or f"configs/{model_name}.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        all_results[model_name] = run_evaluation(model_name, config, prompts)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print_summary(all_results)
    print(f"Full results saved to: {args.output}")


if __name__ == "__main__":
    main()
