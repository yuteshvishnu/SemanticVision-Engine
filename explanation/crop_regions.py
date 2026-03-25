"""
Crop predicted regions from images using segmentation masks.
These crops are the visual input to the LLM explanation module (Phase 3).

Usage (standalone):
    from explanation.crop_regions import crop_from_mask, overlay_mask_on_image
    from PIL import Image
    import numpy as np

    image = Image.open("dog.jpg").convert("RGB")
    mask  = np.load("mask.npy")   # (H, W) binary array

    crop    = crop_from_mask(image, mask, padding=10)
    overlay = overlay_mask_on_image(image, mask)
"""

import numpy as np
from PIL import Image


# ── Core crop ─────────────────────────────────────────────────────────────────

def mask_to_binary(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert a soft mask (logits or probabilities) to a binary mask.

    Args:
        mask:      (H, W) float array
        threshold: cutoff value

    Returns:
        binary: (H, W) uint8 array with values {0, 1}
    """
    return (mask >= threshold).astype(np.uint8)


def crop_from_mask(image: Image.Image, mask: np.ndarray,
                   padding: int = 10) -> Image.Image | None:
    """
    Crop the tight bounding box of a binary mask from an image.
    Adds a small padding so the LLM gets context around the region.

    Args:
        image:   PIL Image
        mask:    (H, W) binary numpy array {0, 1}
        padding: pixels of context to add around the bounding box

    Returns:
        PIL Image of the cropped region,
        or None if the mask is empty (no region detected)
    """
    mask_bool = mask.astype(bool)

    if not mask_bool.any():
        return None

    rows = np.where(mask_bool.any(axis=1))[0]
    cols = np.where(mask_bool.any(axis=0))[0]

    y1 = max(0,             rows[0]  - padding)
    y2 = min(image.height,  rows[-1] + padding)
    x1 = max(0,             cols[0]  - padding)
    x2 = min(image.width,   cols[-1] + padding)

    return image.crop((x1, y1, x2, y2))


# ── Visualization helpers ─────────────────────────────────────────────────────

def overlay_mask_on_image(image: Image.Image, mask: np.ndarray,
                           color: tuple = (255, 80, 80),
                           alpha: float = 0.45) -> Image.Image:
    """
    Blend a colored mask over the original image for visualization.

    Args:
        image: PIL Image
        mask:  (H, W) binary numpy array {0, 1}
        color: RGB tuple for the mask highlight color
        alpha: overlay transparency (0 = invisible, 1 = solid)

    Returns:
        PIL Image with mask blended in
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
    return Image.fromarray(result.astype(np.uint8))


def draw_bounding_box(image: Image.Image, mask: np.ndarray,
                      color: tuple = (255, 80, 80),
                      linewidth: int = 3) -> Image.Image:
    """
    Draw the bounding box of the mask region on the image.
    Useful for quick visual debugging without a full overlay.

    Args:
        image:     PIL Image
        mask:      (H, W) binary numpy array
        color:     RGB border color
        linewidth: thickness of the bounding box lines in pixels

    Returns:
        PIL Image with bounding box drawn
    """
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return image

    rows = np.where(mask_bool.any(axis=1))[0]
    cols = np.where(mask_bool.any(axis=0))[0]

    y1, y2 = int(rows[0]),  int(rows[-1])
    x1, x2 = int(cols[0]),  int(cols[-1])

    img_array = np.array(image.convert("RGB")).copy()

    for lw in range(linewidth):
        # Top and bottom edges
        if y1 + lw < img_array.shape[0]:
            img_array[y1 + lw, x1:x2] = color
        if y2 - lw >= 0:
            img_array[y2 - lw, x1:x2] = color
        # Left and right edges
        if x1 + lw < img_array.shape[1]:
            img_array[y1:y2, x1 + lw] = color
        if x2 - lw >= 0:
            img_array[y1:y2, x2 - lw] = color

    return Image.fromarray(img_array)


# ── Batch helper ──────────────────────────────────────────────────────────────

def crop_batch(images: list, masks: list, padding: int = 10) -> list:
    """
    Crop regions from a list of (image, mask) pairs.

    Args:
        images:  list of PIL Images
        masks:   list of (H, W) binary numpy arrays
        padding: bounding box padding in pixels

    Returns:
        list of PIL Image crops (None where mask was empty)
    """
    return [crop_from_mask(img, msk, padding) for img, msk in zip(images, masks)]
