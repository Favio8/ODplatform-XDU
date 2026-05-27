# Serving Models

This directory is reserved for YOLO segmentation checkpoints that are approved for Web inference.

Recommended local files:

- `yolo26m_seg_best.pt`: default Web analysis model and current best room segmentation checkpoint.
- `room-yolo26n-seg-best.pt`: lightweight room segmentation model for fast demos.
- `room-yolo11m-seg-best.pt`: larger room segmentation model for higher-quality comparison.

Model weights are ignored by git. Copy selected checkpoints from `models/checkpoints/` into this directory before running the Web analysis service.
