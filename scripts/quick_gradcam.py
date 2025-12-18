#!/usr/bin/env python3
"""
Quick Grad-CAM for Single Image (EfficientNet-B4 and B7 only for speed)

Usage:
    python quick_gradcam.py /path/to/image.jpg
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ================================================================================
# Grad-CAM Class
# ================================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[:, i]
        heatmap = F.relu(torch.mean(self.activations, dim=1).squeeze())
        if torch.max(heatmap) > 0:
            heatmap = heatmap / torch.max(heatmap)
        return heatmap.cpu().numpy(), target_class, output

# ================================================================================
# Main Processing
# ================================================================================

def process_image(img_path, save_path):
    print(f"Processing: {img_path}")

    # Load image
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    img_np = np.array(img.resize((384, 384)))

    # Class names
    classes = ['ADM', 'Basal_cell_carcinoma', 'Ephelis', 'Malignant_melanoma',
               'Melasma', 'Nevus', 'Seborrheic_keratosis', 'Solar_lentigo']

    # Collect predictions and heatmaps
    all_probs = []
    all_heatmaps = []

    # Model configurations
    base_dir = Path(__file__).parent.parent.parent.parent
    configs = [
        ('efficientnet_b4', base_dir / 'ensemble_7models/models_ham10000_pretrained/efficientnet_b4_ham10k_fold{}.pth', 'B4-HAM10K'),
        ('efficientnet_b7', base_dir / 'model_92.34/models/improved_efficientnet_b7_fold{}.pth', 'B7-ImageNet')
    ]

    print("\nLoading models and generating Grad-CAM...")

    for arch, path_pattern, name in configs:
        for fold in range(5):
            model_path = Path(str(path_pattern).format(fold))

            if not model_path.exists():
                print(f"  ⚠️  {name} Fold {fold}: Not found")
                continue

            try:
                # Load model
                model = timm.create_model(arch, pretrained=False, num_classes=8)
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

                model.to(device)
                model.eval()

                # Get prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    all_probs.append(probs.cpu().numpy())

                # Generate Grad-CAM
                target_layer = model.conv_head
                gradcam = GradCAM(model, target_layer)

                # Use ensemble prediction as target
                avg_probs = np.mean(all_probs, axis=0)
                pred_class = avg_probs.argmax()

                heatmap, _, _ = gradcam.generate_cam(input_tensor, target_class=pred_class)
                heatmap_resized = cv2.resize(heatmap, (384, 384))
                all_heatmaps.append(heatmap_resized)

                print(f"  ✅ {name} Fold {fold}")

            except Exception as e:
                print(f"  ❌ {name} Fold {fold}: {e}")

    # Calculate ensemble prediction
    avg_probs = np.mean(all_probs, axis=0)
    pred_class = avg_probs.argmax()
    confidence = avg_probs[0, pred_class] * 100

    # Average heatmaps
    avg_heatmap = np.mean(all_heatmaps, axis=0)
    if np.max(avg_heatmap) > 0:
        avg_heatmap = avg_heatmap / np.max(avg_heatmap)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Heatmap
    im = axes[1].imshow(avg_heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    heatmap_color = cv2.applyColorMap(np.uint8(255 * avg_heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.clip(heatmap_color * 0.4 + img_np * 0.6, 0, 255).astype(np.uint8)

    axes[2].imshow(overlay)
    axes[2].set_title(f'Prediction: {classes[pred_class]}\nConfidence: {confidence:.2f}%',
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved: {save_path}")
    print(f"   Prediction: {classes[pred_class]} ({confidence:.2f}%)")
    print(f"   Models used: {len(all_probs)} ({len(all_heatmaps)} heatmaps)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_gradcam.py /path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]

    if not os.path.exists(img_path):
        print(f"❌ Error: Image not found at {img_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'gradcam_results'
    output_dir.mkdir(exist_ok=True)

    save_path = output_dir / f"quick_gradcam_{os.path.basename(img_path)}"

    print("="*80)
    print("Quick Grad-CAM (EfficientNet-B4 + B7 only)")
    print("="*80)

    process_image(img_path, save_path)

    print("\n" + "="*80)
    print("✅ Completed!")
    print("="*80)
