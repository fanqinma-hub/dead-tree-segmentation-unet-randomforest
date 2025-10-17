# Dead Tree Segmentation in Aerial Imagery 🌳

Semantic segmentation of dead trees in UAV imagery using **U-Net (with lightweight attention)** and a **Random Forest baseline**.

## Features
- NDVI/texture features, augmentation, Dice/Focal loss
- Metrics: mIoU / Precision / Recall (confusion matrix)
- Reproducible scripts + clean repo layout

## Structure
- `src/`  Model & training code  
- `notebooks/`  Experiments & visualization  
- `configs/`  Hyperparameters  
- `results/`  Curves / sample predictions  
- `checkpoints/`  (ignored)  
- `data/`  (ignored)

## Run
```bash
pip install -r requirements.txt
python src/train.py --config configs/config.yaml
python src/infer.py --config configs/config.yaml --ckpt checkpoints/best_model.pth
