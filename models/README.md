# Model Weights Directory

This directory stores pretrained weights for **AgriWeed AI**.

## Recommended Default Model
- **Model Name**: `NvMayMay/weedblaster-vision-yolov8s`
- **File Name**: `weedblaster_yolov8s.pt` (or `best.pt`)
- **Dataset**: CropAndWeed Dataset (Steininger et al., WACV 2023)
- **Architecture**: Ultralytics YOLOv8-small
- **Supported Classes**:
  1. `Maize` (Crop)
  2. `Sugar Beet` (Crop)
  3. `Soy` (Crop)
  4. `Sunflower` (Crop)
  5. `Potato` (Crop)
  6. `Pea` (Crop)
  7. `Bean` (Crop)
  8. `Pumpkin` (Crop)
  9. `Weed` (Target Weed Superclass, aggregating 58 weed species)
- **License**: AGPL-3.0
- **Direct Download Link**: [Hugging Face NvMayMay/weedblaster-vision-yolov8s](https://huggingface.co/NvMayMay/weedblaster-vision-yolov8s/resolve/main/best.pt)

## Automatic Setup
AgriWeed AI includes an automated downloader in `ai/model_loader.py` that can retrieve the tested weights file directly into this directory, or you can place any compatible Ultralytics YOLO `.pt` model weights file directly in this folder.
