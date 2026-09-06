# DenWide-Net: Dense WideResNet for Autism Spectrum Disorder (ASD) Detection using Brain MRI Scans

An end-to-end deep learning framework implementing **DenWide-Net (Dense WideResNet)** for computer-aided diagnosis of Autism Spectrum Disorder (ASD) from structural brain Magnetic Resonance Imaging (MRI) scans.

---

## 📌 Scientific Background & Reference

This repository is an implementation inspired by the research:
> **"DenWide-Net: Dense WideresNet for autism spectrum disorder detection using MRI image"**  
> *A. Arunkumar & S. Sudhakar Ilango*  
> Published in *Biomedical Signal Processing and Control* (2026).

### Key Architectural Concepts
1. **Dense Feature Reuse (DenseNet Heritage):** Each layer receives feature representations from all preceding layers within a dense block, mitigating vanishing gradients and maximizing multi-scale biomarker capture.
2. **Channel Capacity Expansion (WideResNet Heritage):** Introduces widened residual units ($k \ge 2$) with residual shortcuts, capturing rich high-dimensional spatial variations in cortical thickness, ventricular asymmetry, and white matter density.
3. **Transition Layers:** $1 \times 1$ convolutions with compression factor $\theta = 0.5$ and average pooling to prevent parameter explosion.
4. **Spatial Attention Gates:** Focuses computational attention on key neuroanatomical structures (e.g., corpus callosum, ventricles, cerebellar vermis).
5. **Grad-CAM Explainability:** Produces visual saliency heatmaps highlighting the exact brain regions influencing model decisions.

---

## 🗂️ Project Structure

```text
d:\mri\
├── config.py                  # Global hyperparameters, dimensions, and path configs
├── requirements.txt           # Python dependencies (PyTorch, Torchvision, etc.)
├── train.py                   # Model training loop with validation & checkpointing
├── evaluate.py                # Comprehensive test set evaluation & metric reporting
├── predict.py                 # Single/Batch scan inference with Grad-CAM visualization
├── app.py                     # Interactive Flask web diagnostic dashboard
├── README.md                  # Project documentation & reference
│
├── models/
│   ├── __init__.py
│   ├── components.py          # Dense-Wide Blocks, Wide Units, Spatial Attention
│   └── denwidenet.py          # Complete 2D & 3D DenWide-Net network architectures
│
├── dataset/
│   ├── __init__.py
│   ├── abide_dataset.py       # ABIDE I / II dataset loader & PyTorch DataLoaders
│   ├── mri_preprocessor.py    # Intensity normalization, slice extraction, transforms
│   └── generate_sample_data.py# Synthetic brain MRI data generator
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py             # Accuracy, Sensitivity, Specificity, F1, ROC-AUC
│   ├── visualization.py       # Grad-CAM, ROC curve, and confusion matrix plots
│   └── optimizer.py           # AdamW / Cosine Annealing learning rate schedulers
│
├── data/                      # Dataset directory (train, val, test)
│   ├── train/ (ASD & TC)
│   ├── val/   (ASD & TC)
│   └── test/  (ASD & TC)
│
├── checkpoints/               # Saved model weights (denwidenet_best.pth)
├── outputs/                   # Training curves, confusion matrix, and Grad-CAM heatmaps
├── templates/
│   └── index.html             # Web dashboard GUI
└── static/
    ├── uploads/               # Temporary scan uploads
    └── results/               # Output Grad-CAM overlays
```

---

## 🚀 Quickstart Guide

### 1. Requirements Installation
```bash
pip install -r requirements.txt
```

### 2. Generate or Load Brain MRI Dataset
The project includes a realistic synthetic T1-weighted brain MRI generator for immediate testing:
```bash
python dataset/generate_sample_data.py --num_train 80 --num_val 20 --num_test 20
```

To use real **ABIDE** scans (Autism Brain Imaging Data Exchange):
Place your preprocessed structural MRI slices or volumes into:
- `data/train/ASD/` and `data/train/TC/`
- `data/val/ASD/` and `data/val/TC/`
- `data/test/ASD/` and `data/test/TC/`

### 3. Train DenWide-Net
```bash
python train.py --epochs 30 --batch_size 16 --lr 0.0001 --widen_factor 2 --growth_rate 24
```
Outputs saved:
- Best model weights: `checkpoints/denwidenet_best.pth`
- Learning curves: `outputs/training_curves.png`

### 4. Evaluate on Test Dataset
```bash
python evaluate.py --model_path checkpoints/denwidenet_best.pth
```
Computes and reports:
- **Accuracy**
- **Sensitivity** (Detection rate for ASD)
- **Specificity** (Ability to correctly rule out ASD)
- **Precision & F1-Score**
- **ROC-AUC** & saves `outputs/confusion_matrix.png` and `outputs/roc_curve.png`

### 5. Run Single-Scan Prediction with Grad-CAM
```bash
python predict.py --image_path data/test/ASD/asd_subject_0001.png
```
Produces classification probabilities and saves the explainability heatmap to `outputs/gradcam_*.png`.

### 6. Launch Web Diagnostic Interface
```bash
python app.py
```
Open your browser at `http://localhost:5000` to interactively upload brain MRI scans, view real-time ASD probability scores, and visualize neurological attention heatmaps.
