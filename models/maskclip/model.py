"""
MaskCLIP — Reimplementation
Paper: Extract Free Dense Labels from CLIP (MaskCLIP)
Reference: https://arxiv.org/abs/2112.01071

Key idea:
    Standard CLIP pools patch tokens into a single global [CLS] embedding.
    MaskCLIP avoids this — it reads the raw patch tokens directly and computes
    cosine similarity against text embeddings to produce a spatial similarity
    map, which becomes the segmentation mask. Zero trainable parameters.

Architecture:
    Image → CLIP ViT → patch tokens (H'×W'×D)
                              ↓
              cosine sim with text embeddings
                              ↓
              similarity map (H'×W'×N_prompts)
                              ↓
              bilinear upsample → mask (H×W×N_prompts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class MaskCLIP(nn.Module):
    def __init__(self, clip_backbone: str = "ViT-B-16", pretrained: str = "openai"):
        super().__init__()

        # Load pretrained CLIP — entire backbone stays frozen
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_backbone, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_backbone)

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.patch_size = 16   # ViT-B/16 → 16px patches
        self.embed_dim  = 512  # ViT-B output dimension

    # ── Text encoding ────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, prompts: list) -> torch.Tensor:
        """
        Encode a list of text prompts into normalized CLIP embeddings.

        Args:
            prompts: e.g. ["a dog on grass", "a cat on a sofa"]

        Returns:
            text_feats: (N, embed_dim) normalized float tensor
        """
        tokens     = self.tokenizer(prompts)
        text_feats = self.clip_model.encode_text(tokens)
        return F.normalize(text_feats, dim=-1)

    # ── Patch feature extraction ─────────────────────────────────────────────

    @torch.no_grad()
    def encode_image_patches(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract per-patch features from the CLIP ViT visual encoder.
        Bypasses the final [CLS] pooling so we keep spatial information.

        Args:
            images: (B, 3, H, W) preprocessed image tensor

        Returns:
            patch_feats: (B, n_patches, embed_dim)

        How it works:
            open_clip's ViT stores intermediate outputs in
            self.clip_model.visual.transformer.
            We hook into the output of the final transformer block,
            slice off the [CLS] token (index 0), and keep the patch tokens.
        """
        visual    = self.clip_model.visual
        B         = images.shape[0]

        # Patchify + positional embedding
        x = visual.conv1(images)                           # (B, D, H', W')
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1) # (B, n_patches, D)

        # Prepend [CLS] token and add positional embeddings
        cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)               # (B, 1+n_patches, D)
        x = x + visual.positional_embedding

        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = visual.transformer(x)                          # (B, 1+n_patches, D)
        x = visual.ln_post(x)

        # Drop [CLS], keep only patch tokens
        patch_feats = x[:, 1:, :]                          # (B, n_patches, D)

        # Project to final embedding space (same projection CLIP uses for [CLS])
        if visual.proj is not None:
            patch_feats = patch_feats @ visual.proj

        return patch_feats                                  # (B, n_patches, embed_dim)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor, prompts: list) -> torch.Tensor:
        """
        Run open-vocabulary segmentation for a batch of images.

        Args:
            images:  (B, 3, H, W) preprocessed image tensor
            prompts: list of N text prompts

        Returns:
            masks: (B, N, H, W) cosine similarity map per prompt.
                   Values in [-1, 1] — threshold at 0 for binary mask.
        """
        B, _, H, W = images.shape
        n_h = H // self.patch_size
        n_w = W // self.patch_size

        # (B, n_patches, D)
        patch_feats = self.encode_image_patches(images)
        patch_feats = F.normalize(patch_feats, dim=-1)

        # (N, D)
        text_feats = self.encode_text(prompts)

        # Cosine similarity: (B, N, n_patches)
        similarity = torch.einsum("bpd,nd->bnp", patch_feats, text_feats)

        # Reshape to spatial grid: (B, N, n_h, n_w)
        masks = similarity.reshape(B, len(prompts), n_h, n_w)

        # Upsample to original image resolution
        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)

        return masks  # (B, N, H, W)
