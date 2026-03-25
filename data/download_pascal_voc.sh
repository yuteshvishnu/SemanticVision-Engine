#!/bin/bash
# Download and extract PASCAL VOC 2012 dataset (~2GB)
#
# Usage:
#   bash data/download_pascal_voc.sh
#
# After running, your data folder will look like:
#   data/pascal_voc/VOCdevkit/VOC2012/
#     ├── JPEGImages/          ← input images (17,125 total)
#     ├── SegmentationClass/   ← ground truth masks (2,913 annotated)
#     └── ImageSets/Segmentation/
#           ├── train.txt
#           └── val.txt        ← 1,449 validation image IDs

set -e  # exit immediately on any error

DATA_DIR="data/pascal_voc"
VOC_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
TAR_FILE="$DATA_DIR/VOCtrainval_11-May-2012.tar"

# ── Check if already downloaded ───────────────────────────────────────────────
if [ -d "$DATA_DIR/VOCdevkit/VOC2012/JPEGImages" ]; then
    echo "[VOC] Dataset already exists at $DATA_DIR/VOCdevkit/VOC2012/"
    echo "[VOC] Delete the folder and re-run this script to re-download."
    exit 0
fi

mkdir -p "$DATA_DIR"

# ── Download ──────────────────────────────────────────────────────────────────
echo "[VOC] Downloading PASCAL VOC 2012 (~2GB)..."
echo "[VOC] Source: $VOC_URL"
wget --progress=bar:force \
     --tries=3 \
     --timeout=60 \
     -O "$TAR_FILE" \
     "$VOC_URL"

# ── Extract ───────────────────────────────────────────────────────────────────
echo ""
echo "[VOC] Extracting..."
tar -xf "$TAR_FILE" -C "$DATA_DIR"
rm "$TAR_FILE"

# ── Verify ────────────────────────────────────────────────────────────────────
IMG_COUNT=$(ls "$DATA_DIR/VOCdevkit/VOC2012/JPEGImages/" | wc -l)
MSK_COUNT=$(ls "$DATA_DIR/VOCdevkit/VOC2012/SegmentationClass/" | wc -l)
VAL_COUNT=$(wc -l < "$DATA_DIR/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt")

echo ""
echo "[VOC] Done."
echo "      Images:     $IMG_COUNT files in JPEGImages/"
echo "      GT masks:   $MSK_COUNT files in SegmentationClass/"
echo "      Val set:    $VAL_COUNT image IDs in ImageSets/Segmentation/val.txt"
echo ""
echo "[VOC] Set dataset.root in your configs to:"
echo "      data/pascal_voc/VOCdevkit/VOC2012/"
