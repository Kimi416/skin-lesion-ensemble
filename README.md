# 7-Model Ensemble for Skin Lesion Classification

8クラス皮膚病変分類のための7モデルアンサンブルシステム（精度: **95.92%**）

## 概要

35個のディープラーニングモデル（7種類 × 5-fold CV）を組み合わせた高精度な皮膚病変診断支援システムです。

### 対象疾患（8クラス）

1. ADM（光線性角化症）
2. Basal cell carcinoma（基底細胞癌）
3. Ephelis（雀卵斑）
4. Malignant melanoma（悪性黒色腫）
5. Melasma（肝斑）
6. Nevus（色素性母斑）
7. Seborrheic keratosis（脂漏性角化症）
8. Solar lentigo（日光黒子）

## 性能

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **95.92%** |
| Macro Precision | 96.09% |
| Macro Recall | 95.56% |
| Macro F1-Score | 95.77% |
| Melanoma Sensitivity | **100%** |
| Melanoma Specificity | **100%** |

## モデル構成

### Base Models（20モデル）
| Model | Pre-training | Accuracy (5-fold CV) |
|-------|-------------|---------------------|
| Swin Transformer × 5 | ImageNet | 94.13% |
| Vision Transformer × 5 | ImageNet | 92.31% |
| EfficientNet-B7 × 5 | ImageNet | 91.73% |
| EfficientNet-B4 × 5 | ImageNet → HAM10000 | 82.17% |

### Binary Classifiers（15モデル）
- ADM vs Nevus × 5
- Nevus vs Melanoma × 5
- Solar vs Ephelis × 5

## セットアップ

### 1. 環境構築

```bash
# リポジトリのクローン
git clone https://github.com/Kimi416/skin-lesion-ensemble.git
cd skin-lesion-ensemble

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. モデルファイルのダウンロード

モデルファイル（約6.4GB）はHugging Faceからダウンロードしてください。

**Hugging Face**: https://huggingface.co/kindai-derma-ai/skin-lesion-ensemble

```bash
# 方法1: huggingface_hubを使用（推奨）
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('kindai-derma-ai/skin-lesion-ensemble', local_dir='models')"

# 方法2: Git LFSを使用
git lfs install
git clone https://huggingface.co/kindai-derma-ai/skin-lesion-ensemble models
```

ダウンロード後のフォルダ構成：
```
models/
├── efficientnet_b4_ham10k/      # EfficientNet-B4 (HAM10000) × 5
├── efficientnet_b7_imagenet/    # EfficientNet-B7 × 5
├── transformers/                 # ViT × 5, Swin × 5
└── binary_classifiers/          # Binary classifiers × 15
```

## 使用方法

### 単一画像の推論

```python
from predict import predict_single_image

result = predict_single_image("path/to/image.jpg")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All probabilities: {result['probabilities']}")
```

### バッチ推論

```bash
python scripts/evaluate_7model_ensemble.py --data_dir path/to/images
```

### Grad-CAM可視化

```bash
python scripts/quick_gradcam.py --image path/to/image.jpg --output gradcam_result.jpg
```

## ファイル構成

```
.
├── README.md                    # このファイル
├── requirements.txt             # 依存パッケージ
├── predict.py                   # 推論用メインスクリプト
├── models/                      # モデルファイル（要ダウンロード）
├── scripts/
│   ├── evaluate_7model_ensemble.py  # 評価スクリプト
│   ├── comprehensive_evaluation.py  # 詳細評価
│   ├── quick_gradcam.py             # Grad-CAM可視化
│   └── train_efficientnet_b4_ham10000.py  # 学習スクリプト
├── docs/
│   ├── MATERIALS_AND_METHODS.md     # 論文用マテメソ
│   └── PAPER_METHODS_AND_RESULTS.md # 論文用結果
└── results/                     # 評価結果（再生成可能）
```

## 動作環境

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (GPU使用時) / Apple Silicon (MPS)
- RAM: 16GB以上推奨
- GPU VRAM: 8GB以上推奨

### 動作確認済み環境
- macOS 14.x (Apple Silicon M1/M2/M3)
- Ubuntu 22.04 + CUDA 12.1
- Windows 11 + CUDA 11.8

## 引用

このコードを使用した場合は、以下を引用してください：

```bibtex
@misc{skin_lesion_ensemble_2024,
  title={7-Model Ensemble for 8-Class Skin Lesion Classification},
  author={Kindai Derma AI Lab},
  year={2024},
  url={https://github.com/Kimi416/skin-lesion-ensemble}
}
```

## ライセンス

研究目的での使用に限ります。商用利用の際はご連絡ください。

## 謝辞

- HAM10000 Dataset
- timm library
- PyTorch

---

**作成日**: 2024年12月
**最終更新**: 2024年12月
