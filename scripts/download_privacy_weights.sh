#!/bin/bash

# Download pretrained weights for privacy-preserving methods
# Grounding DINO and SAM are used for postprocessing privacy detection

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
WEIGHTS_DIR="$ROOT_DIR/weights"

echo "============================================"
echo "Downloading Privacy Detection Weights"
echo "============================================"

# Create weights directory
mkdir -p "$WEIGHTS_DIR"
cd "$WEIGHTS_DIR"

# Download Grounding DINO weights (664 MB)
echo ""
echo "[1/2] Downloading Grounding DINO SwinT-OGC weights..."
if [ -f "groundingdino_swint_ogc.pth" ]; then
    echo "  groundingdino_swint_ogc.pth already exists, skipping"
else
    wget -q --show-progress \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
        -O groundingdino_swint_ogc.pth
    echo "  Downloaded groundingdino_swint_ogc.pth"
fi

# Download SAM ViT-H weights (2.4 GB)
echo ""
echo "[2/2] Downloading SAM ViT-H weights..."
if [ -f "sam_vit_h_4b8939.pth" ]; then
    echo "  sam_vit_h_4b8939.pth already exists, skipping"
else
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
        -O sam_vit_h_4b8939.pth
    echo "  Downloaded sam_vit_h_4b8939.pth"
fi

echo ""
echo "============================================"
echo "All weights downloaded successfully!"
echo "============================================"
echo ""
echo "Downloaded files:"
ls -lh "$WEIGHTS_DIR"/*.pth
echo ""
echo "Total size:"
du -sh "$WEIGHTS_DIR"
echo ""
echo "These weights are used for:"
echo "  - Grounding DINO: Open-vocabulary object detection with text prompts"
echo "  - SAM: Pixel-perfect segmentation masks"
echo "  - Privacy methods D, E, F (postprocessing mode)"
echo ""
