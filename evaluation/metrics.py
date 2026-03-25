"""
Evaluation metrics for semantic grounding comparison.

Four metrics:
    1. IoU / mIoU           — standard mask accuracy vs ground truth
    2. CLIP Similarity       — semantic alignment between predicted region crop and text
    3. Robustness Variance   — IoU stability across prompt paraphrases
    4. Explanation Alignment — how well the LLM explanation matches the original prompt
"""

import torch
import torch.nn.functional as F
import numpy as np
import open_clip


# ── 1. IoU / mIoU ────────────────────────────────────────────────────────────

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Intersection over Union between a predicted and ground truth mask.

    Args:
        pred_mask: (H, W) float array — raw logits or [0,1] probabilities
        gt_mask:   (H, W) binary array {0, 1}
        threshold: binarization cutoff for pred_mask

    Returns:
        iou: float in [0, 1]
    """
    pred_bin = (pred_mask >= threshold).astype(np.float32)
    gt_bin   = gt_mask.astype(np.float32)

    intersection = (pred_bin * gt_bin).sum()
    union        = pred_bin.sum() + gt_bin.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_miou(pred_masks: list, gt_masks: list, threshold: float = 0.5) -> float:
    """
    Mean IoU over a list of (pred, gt) mask pairs.

    Args:
        pred_masks: list of (H, W) float arrays
        gt_masks:   list of (H, W) binary arrays

    Returns:
        miou: float in [0, 1]
    """
    ious = [compute_iou(p, g, threshold) for p, g in zip(pred_masks, gt_masks)]
    return float(np.mean(ious))


# ── 2. CLIP Similarity ───────────────────────────────────────────────────────

class CLIPSimilarityScorer:
    """
    Computes cosine similarity between a cropped region and a text prompt
    using CLIP embeddings. Load once, score many pairs.

    Higher score = predicted region is semantically closer to the prompt.
    This catches cases where IoU is low but the model found a reasonable region.
    """

    def __init__(self, backbone: str = "ViT-B-16", pretrained: str = "openai", device: str = "cpu"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(backbone)
        self.model.eval().to(device)

    @torch.no_grad()
    def score(self, image_crop, text_prompt: str) -> float:
        """
        Args:
            image_crop:  PIL Image of the predicted region (from crop_regions.py)
            text_prompt: the original segmentation query string

        Returns:
            similarity: float in [-1, 1] — higher means better semantic match
        """
        img_tensor   = self.preprocess(image_crop).unsqueeze(0).to(self.device)
        text_tokens  = self.tokenizer([text_prompt]).to(self.device)

        img_feats  = F.normalize(self.model.encode_image(img_tensor), dim=-1)
        text_feats = F.normalize(self.model.encode_text(text_tokens), dim=-1)

        return float((img_feats * text_feats).sum())

    def score_batch(self, image_crops: list, text_prompts: list) -> list:
        """
        Score multiple (crop, prompt) pairs at once.

        Args:
            image_crops:  list of PIL Images
            text_prompts: list of strings, same length as image_crops

        Returns:
            scores: list of floats
        """
        return [self.score(img, txt) for img, txt in zip(image_crops, text_prompts)]


# ── 3. Prompt Robustness Variance ────────────────────────────────────────────

def compute_robustness_variance(iou_scores: dict) -> dict:
    """
    Quantify how stable a model's predictions are across prompt paraphrases.

    High variance = brittle to rephrasing (bad for real-world use).
    Low variance  = robust to rephrasing (good).

    Args:
        iou_scores: dict mapping prompt type to IoU score, e.g.
                    {"base": 0.72, "paraphrase": 0.68, "abstract": 0.51, "vague": 0.44}

    Returns:
        dict with:
            mean_iou       — average IoU across all phrasings
            variance       — spread of IoU scores (lower is better)
            std            — standard deviation
            drop_from_base — worst-case degradation vs the base prompt
            scores         — original input dict echoed back
    """
    scores = list(iou_scores.values())
    base   = iou_scores.get("base", scores[0])

    return {
        "mean_iou":       round(float(np.mean(scores)), 4),
        "variance":       round(float(np.var(scores)), 4),
        "std":            round(float(np.std(scores)), 4),
        "drop_from_base": round(float(base - min(scores)), 4),
        "scores":         iou_scores,
    }


# ── 4. Explanation Alignment Score ───────────────────────────────────────────

class ExplanationAlignmentScorer:
    """
    Measures how semantically aligned an LLM-generated explanation is
    with the original text prompt, using the CLIP text encoder.

    This is our novel metric — not present in any prior MaskCLIP / DenseCLIP work.

    Intuition: a good explanation should use the same semantic concepts as
    the original prompt. We embed both with CLIP and compute cosine similarity.
    """

    def __init__(self, backbone: str = "ViT-B-16", pretrained: str = "openai", device: str = "cpu"):
        self.device = device
        model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        self.model     = model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(backbone)

    @torch.no_grad()
    def score(self, explanation: str, original_prompt: str) -> float:
        """
        Args:
            explanation:     LLM-generated one-sentence region justification
            original_prompt: the text prompt used for segmentation

        Returns:
            alignment_score: float in [-1, 1]
                             > 0.8  = strongly aligned
                             0.5–0.8 = moderately aligned
                             < 0.5  = poor alignment (model may have hallucinated)
        """
        tokens = self.tokenizer([explanation, original_prompt]).to(self.device)
        feats  = F.normalize(self.model.encode_text(tokens), dim=-1)
        return round(float((feats[0] * feats[1]).sum()), 4)
