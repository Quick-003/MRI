"""
Inference Script for DenWide-Net ASD Diagnosis with Grad-CAM Explainability.

Usage:
    python predict.py --image_path path/to/brain_mri.png --model_path checkpoints/denwidenet_best.pth
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

import config
from models import build_denwidenet
from dataset.mri_preprocessor import load_mri_file, get_mri_transforms
from utils.visualization import GradCAM, overlay_heatmap


def predict_single_scan(model, file_path, device, generate_cam=True, save_cam_path=None):
    """
    Runs ASD prediction on an MRI image or volume slice with Grad-CAM visualization.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"MRI file not found: {path}")

    # Load and normalize raw MRI
    mri_data = load_mri_file(path)
    if mri_data.ndim == 3:
        mri_data = mri_data[:, :, mri_data.shape[2] // 2]

    # Convert to PIL Image for preprocessing
    uint8_img = (mri_data * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img).convert("RGB")

    # Apply transform
    transform = get_mri_transforms(config.IMAGE_SIZE, is_train=False)
    tensor_input = transform(uint8_img).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    pred_class = config.CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    result = {
        "file_path": str(path),
        "prediction_index": pred_idx,
        "prediction_label": pred_class,
        "confidence": confidence,
        "probabilities": {
            config.CLASS_NAMES[0]: float(probs[0]),
            config.CLASS_NAMES[1]: float(probs[1])
        },
        "cam_image_path": None
    }

    # Generate Grad-CAM Attention Heatmap
    if generate_cam:
        target_layer = model.get_cam_target_layer()
        cam_generator = GradCAM(model, target_layer)
        heatmap = cam_generator.generate_heatmap(tensor_input, target_class=pred_idx)

        # Overlay heatmap on resized original MRI
        resized_pil = pil_img.resize(config.IMAGE_SIZE)
        overlay = overlay_heatmap(resized_pil, heatmap, alpha=0.5, colormap="jet")

        if save_cam_path is None:
            save_cam_path = config.OUTPUT_DIR / f"gradcam_{path.stem}.png"
        else:
            save_cam_path = Path(save_cam_path)

        save_cam_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(save_cam_path)
        result["cam_image_path"] = str(save_cam_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict Autism Spectrum Disorder (ASD) from Brain MRI.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input MRI scan (.png, .jpg, .dcm, .nii)")
    parser.add_argument("--model_path", type=str, default=str(config.BEST_MODEL_PATH), help="Path to model weights")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--output_cam", type=str, default=None, help="Path to save Grad-CAM heatmap overlay")
    args = parser.parse_args()

    model_file = Path(args.model_path)
    if not model_file.exists():
        print(f"[Error] Model weights not found at {model_file}. Please train the model first.")
        sys.exit(1)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load Model
    print(f"[Model] Loading DenWide-Net from {model_file}...")
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    saved_config = checkpoint.get("config", {})
    growth_rate = saved_config.get("growth_rate", config.GROWTH_RATE)
    widen_factor = saved_config.get("widen_factor", config.WIDEN_FACTOR)

    model = build_denwidenet(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        growth_rate=growth_rate,
        widen_factor=widen_factor,
        dropout_rate=config.DROPOUT_RATE,
        use_attention=config.USE_ATTENTION
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    # Predict
    res = predict_single_scan(model, args.image_path, device, generate_cam=True, save_cam_path=args.output_cam)

    print("\n================== Prediction Result ==================")
    print(f"  MRI File:    {res['file_path']}")
    print(f"  Diagnosis:   {res['prediction_label']}")
    print(f"  Confidence:  {res['confidence'] * 100:.2f}%")
    print(f"  Probabilities:")
    for cls_name, prob in res["probabilities"].items():
        print(f"    - {cls_name}: {prob * 100:.2f}%")
    if res["cam_image_path"]:
        print(f"  Grad-CAM Heatmap: {res['cam_image_path']}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
