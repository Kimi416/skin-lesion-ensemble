# モデルファイルについて

このフォルダにはモデルファイル（.pth）を配置してください。

## ダウンロード

モデルファイルは容量が大きいため（合計約6.4GB）、Git LFSまたは外部ストレージからダウンロードしてください。

**ダウンロードリンク**: [研究室共有フォルダ / Google Drive]

## フォルダ構成

ダウンロード後、以下の構成で配置してください：

```
models/
├── efficientnet_b4_ham10k/
│   ├── efficientnet_b4_ham10k_fold0.pth
│   ├── efficientnet_b4_ham10k_fold1.pth
│   ├── efficientnet_b4_ham10k_fold2.pth
│   ├── efficientnet_b4_ham10k_fold3.pth
│   └── efficientnet_b4_ham10k_fold4.pth
│
├── efficientnet_b7_imagenet/
│   ├── improved_efficientnet_b7_fold0.pth
│   ├── improved_efficientnet_b7_fold1.pth
│   ├── improved_efficientnet_b7_fold2.pth
│   ├── improved_efficientnet_b7_fold3.pth
│   └── improved_efficientnet_b7_fold4.pth
│
├── transformers/
│   ├── vit_transformer_fold0.pth
│   ├── vit_transformer_fold1.pth
│   ├── vit_transformer_fold2.pth
│   ├── vit_transformer_fold3.pth
│   ├── vit_transformer_fold4.pth
│   ├── swin_transformer_fold0.pth
│   ├── swin_transformer_fold1.pth
│   ├── swin_transformer_fold2.pth
│   ├── swin_transformer_fold3.pth
│   └── swin_transformer_fold4.pth
│
└── binary_classifiers/
    ├── binary_adm_vs_nevus_fold0.pth
    ├── binary_adm_vs_nevus_fold1.pth
    ├── binary_adm_vs_nevus_fold2.pth
    ├── binary_adm_vs_nevus_fold3.pth
    ├── binary_adm_vs_nevus_fold4.pth
    ├── binary_nevus_vs_melanoma_fold0.pth
    ├── binary_nevus_vs_melanoma_fold1.pth
    ├── binary_nevus_vs_melanoma_fold2.pth
    ├── binary_nevus_vs_melanoma_fold3.pth
    ├── binary_nevus_vs_melanoma_fold4.pth
    ├── binary_solar_vs_ephelis_fold0.pth
    ├── binary_solar_vs_ephelis_fold1.pth
    ├── binary_solar_vs_ephelis_fold2.pth
    ├── binary_solar_vs_ephelis_fold3.pth
    └── binary_solar_vs_ephelis_fold4.pth
```

## ファイルサイズ

| モデル | サイズ |
|--------|--------|
| EfficientNet-B4 (HAM10K) × 5 | ~1.0 GB |
| EfficientNet-B7 × 5 | ~1.2 GB |
| Vision Transformer × 5 | ~1.7 GB |
| Swin Transformer × 5 | ~1.4 GB |
| Binary Classifiers × 15 | ~1.1 GB |
| **合計** | **~6.4 GB** |

## チェックサム（SHA256）

ダウンロード後、ファイルの整合性を確認してください：

```bash
# Linux/macOS
sha256sum models/**/*.pth

# Windows
Get-FileHash models\**\*.pth
```

（チェックサムは後で追加予定）
