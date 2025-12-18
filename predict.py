#!/usr/bin/env python3
"""
7-Model Ensemble Prediction Script
単一画像に対する推論を行うスクリプト

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --output result.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

# クラス名の定義
CLASS_NAMES = [
    'ADM',                    # 光線性角化症
    'Basal_cell_carcinoma',   # 基底細胞癌
    'Ephelis',                # 雀卵斑
    'Malignant_melanoma',     # 悪性黒色腫
    'Melasma',                # 肝斑
    'Nevus',                  # 色素性母斑
    'Seborrheic_keratosis',   # 脂漏性角化症
    'Solar_lentigo'           # 日光黒子
]

# クラス名の日本語訳
CLASS_NAMES_JP = {
    'ADM': '光線性角化症',
    'Basal_cell_carcinoma': '基底細胞癌',
    'Ephelis': '雀卵斑',
    'Malignant_melanoma': '悪性黒色腫',
    'Melasma': '肝斑',
    'Nevus': '色素性母斑',
    'Seborrheic_keratosis': '脂漏性角化症',
    'Solar_lentigo': '日光黒子'
}

# デフォルトのモデルパス
DEFAULT_MODEL_DIR = Path(__file__).parent / 'models'


def get_device():
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_transforms():
    """推論用の画像変換を取得"""
    return transforms.Compose([
        transforms.Resize((450, 600)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_efficientnet_b4_ham10k(model_path, device, num_classes=8):
    """EfficientNet-B4 (HAM10000事前学習) をロード"""
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def load_efficientnet_b7(model_path, device, num_classes=8):
    """EfficientNet-B7をロード"""
    model = timm.create_model('efficientnet_b7', pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def load_vit(model_path, device, num_classes=8):
    """Vision Transformerをロード"""
    model = timm.create_model('vit_base_patch16_384', pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def load_swin(model_path, device, num_classes=8):
    """Swin Transformerをロード"""
    model = timm.create_model('swin_base_patch4_window12_384', pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def load_binary_classifier(model_path, device):
    """2値分類器をロード"""
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


class EnsemblePredictor:
    """7モデルアンサンブル予測器"""

    def __init__(self, model_dir=None, device=None):
        self.device = device or get_device()
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.transform = get_transforms()
        self.models_loaded = False

        # モデルを格納
        self.base_models = []
        self.binary_adm_nevus = []
        self.binary_nevus_mm = []
        self.binary_solar_ephelis = []

    def load_models(self):
        """全35モデルをロード"""
        print(f"Loading models from {self.model_dir}...")
        print(f"Device: {self.device}")

        # EfficientNet-B4 (HAM10000) × 5
        b4_dir = self.model_dir / 'efficientnet_b4_ham10k'
        for i in range(5):
            path = b4_dir / f'efficientnet_b4_ham10k_fold{i}.pth'
            if path.exists():
                self.base_models.append(load_efficientnet_b4_ham10k(path, self.device))
                print(f"  ✓ EfficientNet-B4 (HAM10K) fold{i}")

        # EfficientNet-B7 × 5
        b7_dir = self.model_dir / 'efficientnet_b7_imagenet'
        for i in range(5):
            path = b7_dir / f'improved_efficientnet_b7_fold{i}.pth'
            if path.exists():
                self.base_models.append(load_efficientnet_b7(path, self.device))
                print(f"  ✓ EfficientNet-B7 fold{i}")

        # Vision Transformer × 5
        trans_dir = self.model_dir / 'transformers'
        for i in range(5):
            path = trans_dir / f'vit_transformer_fold{i}.pth'
            if path.exists():
                self.base_models.append(load_vit(path, self.device))
                print(f"  ✓ ViT fold{i}")

        # Swin Transformer × 5
        for i in range(5):
            path = trans_dir / f'swin_transformer_fold{i}.pth'
            if path.exists():
                self.base_models.append(load_swin(path, self.device))
                print(f"  ✓ Swin fold{i}")

        # Binary Classifiers
        binary_dir = self.model_dir / 'binary_classifiers'

        # ADM vs Nevus × 5
        for i in range(5):
            path = binary_dir / f'binary_adm_vs_nevus_fold{i}.pth'
            if path.exists():
                self.binary_adm_nevus.append(load_binary_classifier(path, self.device))
                print(f"  ✓ ADM-Nevus fold{i}")

        # Nevus vs Melanoma × 5
        for i in range(5):
            path = binary_dir / f'binary_nevus_vs_melanoma_fold{i}.pth'
            if path.exists():
                self.binary_nevus_mm.append(load_binary_classifier(path, self.device))
                print(f"  ✓ Nevus-MM fold{i}")

        # Solar vs Ephelis × 5
        for i in range(5):
            path = binary_dir / f'binary_solar_vs_ephelis_fold{i}.pth'
            if path.exists():
                self.binary_solar_ephelis.append(load_binary_classifier(path, self.device))
                print(f"  ✓ Solar-Ephelis fold{i}")

        total = len(self.base_models) + len(self.binary_adm_nevus) + \
                len(self.binary_nevus_mm) + len(self.binary_solar_ephelis)
        print(f"\nLoaded {total}/35 models")
        self.models_loaded = True

    def predict(self, image_path):
        """単一画像の予測"""
        if not self.models_loaded:
            self.load_models()

        # 画像の読み込みと前処理
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Stage 1: ベースモデルのアンサンブル
        base_probs = torch.zeros(1, 8).to(self.device)

        with torch.no_grad():
            for model in self.base_models:
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                base_probs += probs

        base_probs /= len(self.base_models)

        # Stage 2: 2値分類器による補正
        refined_probs = base_probs.clone()

        # ADM vs Nevus補正
        if self.binary_adm_nevus:
            binary_probs = torch.zeros(1, 2).to(self.device)
            for model in self.binary_adm_nevus:
                output = model(input_tensor)
                binary_probs += F.softmax(output, dim=1)
            binary_probs /= len(self.binary_adm_nevus)

            # 確率の再配分
            total = refined_probs[0, 0] + refined_probs[0, 5]  # ADM + Nevus
            refined_probs[0, 0] = total * binary_probs[0, 0]   # ADM
            refined_probs[0, 5] = total * binary_probs[0, 1]   # Nevus

        # Nevus vs Melanoma補正
        if self.binary_nevus_mm:
            binary_probs = torch.zeros(1, 2).to(self.device)
            for model in self.binary_nevus_mm:
                output = model(input_tensor)
                binary_probs += F.softmax(output, dim=1)
            binary_probs /= len(self.binary_nevus_mm)

            total = refined_probs[0, 5] + refined_probs[0, 3]  # Nevus + MM
            refined_probs[0, 5] = total * binary_probs[0, 0]   # Nevus
            refined_probs[0, 3] = total * binary_probs[0, 1]   # MM

        # Solar vs Ephelis補正
        if self.binary_solar_ephelis:
            binary_probs = torch.zeros(1, 2).to(self.device)
            for model in self.binary_solar_ephelis:
                output = model(input_tensor)
                binary_probs += F.softmax(output, dim=1)
            binary_probs /= len(self.binary_solar_ephelis)

            total = refined_probs[0, 7] + refined_probs[0, 2]  # Solar + Ephelis
            refined_probs[0, 7] = total * binary_probs[0, 0]   # Solar
            refined_probs[0, 2] = total * binary_probs[0, 1]   # Ephelis

        # 最終予測
        refined_probs = refined_probs.cpu().numpy()[0]
        pred_idx = np.argmax(refined_probs)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = refined_probs[pred_idx]

        return {
            'prediction': pred_class,
            'prediction_jp': CLASS_NAMES_JP[pred_class],
            'confidence': float(confidence),
            'probabilities': {
                CLASS_NAMES[i]: float(refined_probs[i])
                for i in range(8)
            }
        }


def predict_single_image(image_path, model_dir=None):
    """単一画像の予測（便利関数）"""
    predictor = EnsemblePredictor(model_dir=model_dir)
    return predictor.predict(image_path)


def main():
    parser = argparse.ArgumentParser(description='7-Model Ensemble Prediction')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model_dir', default=None, help='Path to models directory')
    parser.add_argument('--output', default=None, help='Path to output JSON file')
    args = parser.parse_args()

    # 予測実行
    result = predict_single_image(args.image, args.model_dir)

    # 結果表示
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Image: {args.image}")
    print(f"Prediction: {result['prediction']} ({result['prediction_jp']})")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nAll Probabilities:")
    for class_name, prob in sorted(result['probabilities'].items(),
                                    key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"  {class_name:25s}: {prob:6.2%} {bar}")
    print("="*50)

    # JSON出力
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResult saved to: {args.output}")


if __name__ == '__main__':
    main()
