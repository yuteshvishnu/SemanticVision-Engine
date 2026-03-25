"""
DenseCLIP — Reimplementation
Paper: DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting
Reference: https://arxiv.org/abs/2112.01518

Key idea:
    Unlike MaskCLIP (zero trainable params), DenseCLIP adds a trainable
    ContextDecoder that uses cross-attention to inject CLIP language features
    into the visual patch tokens before the segmentation head.
    This gives the model stronger spatial grounding, especially for
    abstract or paraphrased prompts.

Architecture:
    Image → CLIP ViT → patch tokens (H'×W'×D)  ← same as MaskCLIP
                              ↓
              ContextDecoder (cross-attn with text) ← trainable
                              ↓
              Conv segmentation head               ← trainable
                              ↓
              bilinear upsample → mask (H×W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


# ── Context Decoder ──────────────────────────────────────────────────────────

class ContextDecoder(nn.Module):
    """
    The core novel component of DenseCLIP.

    Uses multi-head cross-attention to enrich spatial visual features
    with language context. Each visual patch attends to the text embedding,
    pulling in semantic guidance before the segmentation head.

    Query  = visual patch tokens  (what we want to segment)
    Key/Value = text embedding    (what we're looking for)
    """

    def __init__(self, visual_dim: int = 512, text_dim: int = 512, num_heads: int = 4):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            kdim=text_dim,
            vdim=text_dim,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(visual_dim)
        self.ffn  = nn.Sequential(
            nn.Linear(visual_dim, visual_dim * 2),
            nn.GELU(),
            nn.Linear(visual_dim * 2, visual_dim),
        )
        self.norm2 = nn.LayerNorm(visual_dim)

    def forward(self, visual_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feats: (B, n_patches, visual_dim) — spatial patch features from CLIP ViT
            text_feats:   (B, 1, text_dim)           — single text embedding as context

        Returns:
            enriched: (B, n_patches, visual_dim) — language-enriched visual features
        """
        # Cross-attention: patches query the text context
        attn_out, _ = self.cross_attn(
            query=visual_feats,
            key=text_feats,
            value=text_feats,
        )

        # Residual + LayerNorm (Pre-LN style, more stable)
        x = self.norm(visual_feats + attn_out)

        # Feed-forward + residual
        x = self.norm2(x + self.ffn(x))

        return x  # (B, n_patches, visual_dim)


# ── DenseCLIP ─────────────────────────────────────────────────────────────────

class DenseCLIP(nn.Module):
    def __init__(self, clip_backbone: str = "ViT-B-16", pretrained: str = "openai"):
        super().__init__()

        # Load pretrained CLIP — backbone stays frozen
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_backbone, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_backbone)

        for param in self.clip_model.parameters():
            param.requires_grad = False

        embed_dim = 512

        # Trainable components — everything below here gets gradient updates
        self.context_decoder = ContextDecoder(
            visual_dim=embed_dim,
            text_dim=embed_dim,
            num_heads=4,
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),  # binary: foreground vs background
        )

        self.patch_size = 16
        self.embed_dim  = embed_dim

    # ── Text encoding ────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, prompts: list) -> torch.Tensor:
        """
        Encode text prompts into normalized CLIP embeddings.

        Returns:
            text_feats: (N, embed_dim)
        """
        tokens     = self.tokenizer(prompts)
        text_feats = self.clip_model.encode_text(tokens)
        return F.normalize(text_feats, dim=-1)

    # ── Patch feature extraction ─────────────────────────────────────────────

    @torch.no_grad()
    def encode_image_patches(self, images: torch.Tensor) -> torch.Tensor:
        """
        Same patch token extraction as MaskCLIP — bypasses global pooling.

        Returns:
            patch_feats: (B, n_patches, embed_dim)
        """
        visual = self.clip_model.visual
        B      = images.shape[0]

        x = visual.conv1(images)
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)

        cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + visual.positional_embedding

        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = visual.transformer(x)
        x = visual.ln_post(x)

        patch_feats = x[:, 1:, :]  # drop [CLS]

        if visual.proj is not None:
            patch_feats = patch_feats @ visual.proj

        return patch_feats  # (B, n_patches, embed_dim)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor, prompts: list) -> torch.Tensor:
        """
        Run DenseCLIP segmentation for a batch of images.

        For each prompt:
            1. Get frozen CLIP patch features
            2. Enrich with text context via ContextDecoder (trainable)
            3. Pass through conv segmentation head (trainable)
            4. Upsample to original resolution

        Args:
            images:  (B, 3, H, W)
            prompts: list of N text prompts

        Returns:
            masks: (B, N, H, W) — raw logits, apply sigmoid for probabilities
        """
        B, _, H, W = images.shape
        n_h = H // self.patch_size
        n_w = W // self.patch_size

        # Frozen CLIP patch features: (B, n_patches, D)
        patch_feats = self.encode_image_patches(images)

        # Text features: (N, D)
        text_feats = self.encode_text(prompts)

        all_masks = []
        for i in range(len(prompts)):
            # Single text embedding as context: (B, 1, D)
            t = text_feats[i].unsqueeze(0).unsqueeze(0).expand(B, -1, -1)

            # Language-enriched visual features: (B, n_patches, D)
            enriched = self.context_decoder(patch_feats, t)

            # Reshape to 2D spatial feature map: (B, D, n_h, n_w)
            spatial = enriched.permute(0, 2, 1).reshape(B, self.embed_dim, n_h, n_w)

            # Conv head → (B, 1, n_h, n_w)
            mask = self.seg_head(spatial)

            # Upsample to original resolution
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
            all_masks.append(mask)

        return torch.cat(all_masks, dim=1)  # (B, N, H, W)
