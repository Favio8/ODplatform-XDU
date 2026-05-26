# Data Layout

- `raw/<dataset>`: source dataset root, e.g. `data/raw/rsod`
- `yolo/<dataset>`: intermediate YOLO-format labels when a converter needs them
- `train`, `val`, `test`: final split outputs used by training and evaluation

Keep large dataset files outside Git history.
