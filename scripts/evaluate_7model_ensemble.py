#!/usr/bin/env python3
"""
7モデルアンサンブル評価スクリプト

使用モデル:
1. EfficientNet-B4 (HAM10000事前学習) × 5 folds
2. EfficientNet-B7 (ImageNet事前学習) × 5 folds
3. ADM vs Nevus 2値分類器 × 5 folds
4. Nevus vs MM 2値分類器 × 5 folds
5. Solar_lentigo vs Ephelis 2値分類器 × 5 folds
6. Vision Transformer × 5 folds
7. Swin Transformer × 5 folds

合計: 35モデル
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}\n')

# ================================================================================
# データ準備
# ================================================================================

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# データロード
data_dir = Path('/Users/iinuma/Desktop/HAM10000 ver./organized')
class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

all_image_paths = []
all_labels = []

for class_idx, class_name in enumerate(class_names):
    class_dir = data_dir / class_name
    images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
    for img_path in images:
        all_image_paths.append(str(img_path))
        all_labels.append(class_idx)

all_image_paths = np.array(all_image_paths)
all_labels = np.array(all_labels)

_, test_paths, _, test_labels = train_test_split(
    all_image_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

test_dataset = SkinLesionDataset(test_paths, test_labels, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

print(f'Class names: {class_names}')
print(f'Test images: {len(test_paths)}\n')

# ================================================================================
# モデルロード
# ================================================================================

print('='*80)
print('7モデルアンサンブル - モデルロード開始')
print('='*80)

all_models = []
base_dir = Path('/Users/iinuma/Desktop/HAM10000 ver.')

# 1. EfficientNet-B4 (HAM10000事前学習) × 5
print('\n[1/7] Loading EfficientNet-B4 (HAM10000 pretrained)...')
for fold_idx in range(5):
    model_path = base_dir / 'ensemble_7models' / 'models_ham10000_pretrained' / f'efficientnet_b4_ham10k_fold{fold_idx}.pth'
    if model_path.exists():
        model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=len(class_names))
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        all_models.append(('efficientnet_b4_ham10k', model))
        print(f'  ✅ Fold {fold_idx}')
    else:
        print(f'  ❌ Fold {fold_idx} not found: {model_path}')

# 2. EfficientNet-B7 (ImageNet事前学習) × 5
print('\n[2/7] Loading EfficientNet-B7 (ImageNet pretrained)...')
for fold_idx in range(5):
    model_path = base_dir / 'model_92.34' / 'models' / f'improved_efficientnet_b7_fold{fold_idx}.pth'
    if model_path.exists():
        model = timm.create_model('efficientnet_b7', pretrained=False, num_classes=len(class_names))
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        all_models.append(('efficientnet_b7_imagenet', model))
        print(f'  ✅ Fold {fold_idx}')
    else:
        print(f'  ❌ Fold {fold_idx} not found: {model_path}')

# 3. ADM vs Nevus 2値分類器 × 5
print('\n[3/7] Loading ADM vs Nevus binary classifiers...')
binary_adm_nevus_models = []
for fold_idx in range(5):
    model_path = base_dir / 'model_94.90' / 'models' / f'binary_adm_vs_nevus_fold{fold_idx}.pth'
    if model_path.exists():
        model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        binary_adm_nevus_models.append(model)
        print(f'  ✅ Fold {fold_idx}')
    else:
        print(f'  ❌ Fold {fold_idx} not found: {model_path}')

# 4. Nevus vs MM 2値分類器 × 5
print('\n[4/7] Loading Nevus vs Melanoma binary classifiers...')
binary_nevus_mm_models = []
for fold_idx in range(5):
    model_path = base_dir / 'model_94.90' / 'models' / f'binary_nevus_vs_melanoma_fold{fold_idx}.pth'
    if model_path.exists():
        model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        binary_nevus_mm_models.append(model)
        print(f'  ✅ Fold {fold_idx}')
    else:
        print(f'  ❌ Fold {fold_idx} not found: {model_path}')

# 5. Solar vs Ephelis 2値分類器 × 5
print('\n[5/7] Loading Solar_lentigo vs Ephelis binary classifiers...')
binary_solar_ephelis_models = []
for fold_idx in range(5):
    model_path = base_dir / 'model_94.90' / 'models' / f'binary_solar_vs_ephelis_fold{fold_idx}.pth'
    if model_path.exists():
        model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        binary_solar_ephelis_models.append(model)
        print(f'  ✅ Fold {fold_idx}')
    else:
        print(f'  ❌ Fold {fold_idx} not found: {model_path}')

# 6. Vision Transformer × 5
print('\n[6/7] Loading Vision Transformer models...')
for fold_idx in range(5):
    model_path = base_dir / 'model_94.90' / 'models' / f'vit_transformer_fold{fold_idx}.pth'
    if model_path.exists():
        model = timm.create_model('vit_base_patch16_384', pretrained=False, num_classes=len(class_names))
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        all_models.append(('vit', model))
        print(f'  ✅ Fold {fold_idx}')
    else:
        print(f'  ❌ Fold {fold_idx} not found: {model_path}')

# 7. Swin Transformer × 5
print('\n[7/7] Loading Swin Transformer models...')
for fold_idx in range(5):
    model_path = base_dir / 'model_94.90' / 'models' / f'swin_transformer_fold{fold_idx}.pth'
    if model_path.exists():
        model = timm.create_model('swin_base_patch4_window12_384', pretrained=False, num_classes=len(class_names))
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        all_models.append(('swin', model))
        print(f'  ✅ Fold {fold_idx}')
    else:
        print(f'  ❌ Fold {fold_idx} not found: {model_path}')

print(f'\n✅ Base models loaded: {len(all_models)} models')
print(f'✅ ADM-Nevus binary models: {len(binary_adm_nevus_models)} models')
print(f'✅ Nevus-MM binary models: {len(binary_nevus_mm_models)} models')
print(f'✅ Solar-Ephelis binary models: {len(binary_solar_ephelis_models)} models')

# ================================================================================
# 2値分類器の適用ロジック
# ================================================================================

class_to_idx = {name: idx for idx, name in enumerate(class_names)}
adm_idx = class_to_idx.get('Actinic_keratosis', -1)
nevus_idx = class_to_idx.get('Nevus', -1)
melanoma_idx = class_to_idx.get('Malignant_melanoma', -1)
solar_idx = class_to_idx.get('Solar_lentigo', -1)
ephelis_idx = class_to_idx.get('Ephelis', -1)

print(f'\nClass indices:')
print(f'  ADM: {adm_idx}')
print(f'  Nevus: {nevus_idx}')
print(f'  Melanoma: {melanoma_idx}')
print(f'  Solar: {solar_idx}')
print(f'  Ephelis: {ephelis_idx}')

def apply_binary_classifiers(base_probs, images):
    """
    2値分類器で予測を補正
    """
    refined_probs = base_probs.clone()

    # ADM vs Nevus
    if len(binary_adm_nevus_models) > 0 and adm_idx != -1 and nevus_idx != -1:
        with torch.no_grad():
            binary_probs_list = []
            for model in binary_adm_nevus_models:
                output = torch.softmax(model(images), dim=1)
                binary_probs_list.append(output)
            binary_probs = torch.stack(binary_probs_list).mean(dim=0)

            # ADM vs Nevusの予測が強い場合、その確率を反映
            adm_nevus_total = refined_probs[:, adm_idx] + refined_probs[:, nevus_idx]
            refined_probs[:, adm_idx] = adm_nevus_total * binary_probs[:, 0]  # class 0 = ADM
            refined_probs[:, nevus_idx] = adm_nevus_total * binary_probs[:, 1]  # class 1 = Nevus

    # Nevus vs MM
    if len(binary_nevus_mm_models) > 0 and nevus_idx != -1 and melanoma_idx != -1:
        with torch.no_grad():
            binary_probs_list = []
            for model in binary_nevus_mm_models:
                output = torch.softmax(model(images), dim=1)
                binary_probs_list.append(output)
            binary_probs = torch.stack(binary_probs_list).mean(dim=0)

            nevus_mm_total = refined_probs[:, nevus_idx] + refined_probs[:, melanoma_idx]
            refined_probs[:, nevus_idx] = nevus_mm_total * binary_probs[:, 0]  # class 0 = Nevus
            refined_probs[:, melanoma_idx] = nevus_mm_total * binary_probs[:, 1]  # class 1 = MM

    # Solar vs Ephelis
    if len(binary_solar_ephelis_models) > 0 and solar_idx != -1 and ephelis_idx != -1:
        with torch.no_grad():
            binary_probs_list = []
            for model in binary_solar_ephelis_models:
                output = torch.softmax(model(images), dim=1)
                binary_probs_list.append(output)
            binary_probs = torch.stack(binary_probs_list).mean(dim=0)

            solar_ephelis_total = refined_probs[:, solar_idx] + refined_probs[:, ephelis_idx]
            refined_probs[:, solar_idx] = solar_ephelis_total * binary_probs[:, 0]  # class 0 = Solar
            refined_probs[:, ephelis_idx] = solar_ephelis_total * binary_probs[:, 1]  # class 1 = Ephelis

    return refined_probs

# ================================================================================
# 評価
# ================================================================================

print('\n' + '='*80)
print('評価開始')
print('='*80)

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Evaluating'):
        images = images.to(device)

        # 全ベースモデルの予測を集計
        batch_probs_list = []
        for model_name, model in all_models:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            batch_probs_list.append(probs)

        # 平均をとる
        base_ensemble_probs = torch.stack(batch_probs_list).mean(dim=0)

        # 2値分類器で補正
        refined_probs = apply_binary_classifiers(base_ensemble_probs, images)

        # 予測クラス
        _, preds = torch.max(refined_probs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ================================================================================
# 結果出力
# ================================================================================

accuracy = accuracy_score(all_labels, all_preds)
print(f'\n{"="*80}')
print(f'7モデルアンサンブル結果')
print(f'{"="*80}')
print(f'Test Accuracy: {accuracy*100:.2f}%\n')

print('Classification Report:')
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

cm = confusion_matrix(all_labels, all_preds)
print('\nConfusion Matrix:')
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print(cm_df)

# CSV保存
output_dir = Path('ensemble_7models')
output_dir.mkdir(exist_ok=True)

# Confusion Matrix保存
cm_df.to_csv(output_dir / 'confusion_matrix.csv')

# 詳細レポート保存
report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(output_dir / 'classification_report.csv')

# サマリー保存
summary = {
    'Total Models': len(all_models) + len(binary_adm_nevus_models) + len(binary_nevus_mm_models) + len(binary_solar_ephelis_models),
    'Base Models': len(all_models),
    'Binary Classifiers': len(binary_adm_nevus_models) + len(binary_nevus_mm_models) + len(binary_solar_ephelis_models),
    'Test Accuracy': f'{accuracy*100:.2f}%',
    'Test Samples': len(test_labels)
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(output_dir / 'summary.csv', index=False)

print(f'\n✅ Results saved to {output_dir}/')
print('  - confusion_matrix.csv')
print('  - classification_report.csv')
print('  - summary.csv')
