#!/usr/bin/env python3
"""
EfficientNet-B4 with HAM10000 Pre-training → 8-disease Fine-tuning (5-fold CV)

HAM10000事前学習済みモデルを使用して、8疾患分類でファインチューニング
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}\n')

# ================================================================================
# 設定
# ================================================================================

CONFIG = {
    'ham10000_pretrained_path': '../backup/ham10000_pretrained_model.pth',
    'data_dir': '../organized',
    'output_dir': 'models_ham10000_pretrained',
    'num_epochs': 50,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'patience': 15,
    'n_folds': 5,
    'random_state': 42
}

# ================================================================================
# データ拡張
# ================================================================================

train_transform = transforms.Compose([
    transforms.Resize((450, 600)),
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
])

val_transform = transforms.Compose([
    transforms.Resize((450, 600)),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================================================================================
# Dataset
# ================================================================================

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

# ================================================================================
# データロード
# ================================================================================

data_dir = Path(CONFIG['data_dir'])
class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

all_image_paths = []
all_labels = []

print('Loading dataset...')
for class_idx, class_name in enumerate(class_names):
    class_dir = data_dir / class_name
    images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
    for img_path in images:
        all_image_paths.append(str(img_path))
        all_labels.append(class_idx)
    print(f'  {class_name}: {len(images)} images')

all_image_paths = np.array(all_image_paths)
all_labels = np.array(all_labels)

print(f'\nTotal images: {len(all_image_paths)}')
print(f'Number of classes: {len(class_names)}')
print(f'Class names: {class_names}\n')

# ================================================================================
# モデル作成関数
# ================================================================================

def create_model_with_ham10000_pretrain(num_classes, pretrained_path):
    """
    HAM10000事前学習済みモデルをロードして8クラス分類用に準備
    """
    # EfficientNet-B4モデルを作成（ImageNet重みなし）
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)

    # HAM10000事前学習の重みをロード
    if os.path.exists(pretrained_path):
        print(f'\nLoading HAM10000 pretrained weights from: {pretrained_path}')
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)

        # チェックポイントから状態辞書を取得
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # モデルの現在の状態辞書を取得
        model_dict = model.state_dict()

        # 分類層（classifier, head, fc）以外の重みのみをロード
        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and 'classifier' not in k and 'head' not in k and 'fc' not in k
        }

        # 重みを更新
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f'✅ Loaded {len(pretrained_dict)} layers from HAM10000 pretrained model')
        print(f'   (Classifier layer randomly initialized for {num_classes} classes)')
    else:
        print(f'❌ HAM10000 pretrained model not found: {pretrained_path}')
        print('   Using random initialization instead')

    return model.to(device)

# ================================================================================
# Focal Loss
# ================================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ================================================================================
# トレーニング関数
# ================================================================================

def train_one_fold(fold_idx, train_loader, val_loader, model_save_path):
    """
    1つのfoldをトレーニング
    """
    print(f'\n{"="*80}')
    print(f'Fold {fold_idx + 1}/{CONFIG["n_folds"]} - Training Start')
    print(f'{"="*80}')

    # モデル作成
    model = create_model_with_ham10000_pretrain(
        num_classes=len(class_names),
        pretrained_path=CONFIG['ham10000_pretrained_path']
    )

    # 損失関数、オプティマイザ、スケジューラ
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(CONFIG['num_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        print(f'Epoch {epoch+1}/{CONFIG["num_epochs"]} - '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Acc: {val_acc:.4f}')

        # Early stopping & Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'fold': fold_idx,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, model_save_path)
            print(f'  ✅ Best model saved! Val Acc: {val_acc:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f'  ⚠️ Early stopping at epoch {epoch+1}')
                break

        scheduler.step()

    print(f'\nFold {fold_idx + 1} completed. Best Val Acc: {best_val_acc:.4f}')
    return best_val_acc

# ================================================================================
# メイン処理（5-Fold CV）
# ================================================================================

print('='*80)
print('5-Fold Cross-Validation Training with HAM10000 Pre-trained Weights')
print('='*80)

# 出力ディレクトリ作成
output_dir = Path(CONFIG['output_dir'])
output_dir.mkdir(exist_ok=True)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_state'])

fold_results = []

for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(all_image_paths, all_labels)):
    # 80% train + validation, 20% test (testは使わない)
    train_val_paths = all_image_paths[train_val_idx]
    train_val_labels = all_labels[train_val_idx]

    # train_valを80:20でtrainとvalに分割
    split_idx = int(0.8 * len(train_val_paths))
    indices = np.arange(len(train_val_paths))
    np.random.seed(CONFIG['random_state'] + fold_idx)
    np.random.shuffle(indices)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_paths = train_val_paths[train_indices]
    train_labels = train_val_labels[train_indices]
    val_paths = train_val_paths[val_indices]
    val_labels = train_val_labels[val_indices]

    # Dataset & DataLoader
    train_dataset = SkinLesionDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = SkinLesionDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    print(f'\nFold {fold_idx + 1}:')
    print(f'  Train: {len(train_paths)} images')
    print(f'  Val: {len(val_paths)} images')

    # モデル保存パス
    model_save_path = output_dir / f'efficientnet_b4_ham10k_fold{fold_idx}.pth'

    # トレーニング
    best_val_acc = train_one_fold(fold_idx, train_loader, val_loader, model_save_path)
    fold_results.append(best_val_acc)

# ================================================================================
# 結果サマリー
# ================================================================================

print('\n' + '='*80)
print('5-Fold Cross-Validation Results')
print('='*80)

for fold_idx, acc in enumerate(fold_results):
    print(f'Fold {fold_idx + 1}: {acc*100:.2f}%')

mean_acc = np.mean(fold_results)
std_acc = np.std(fold_results)

print(f'\nMean Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%')
print(f'\n✅ All models saved to: {output_dir}/')
print('='*80)
