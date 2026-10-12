# GSLC – Transfer Learning pada Oxford-IIIT Pet

Eksperimen perbandingan strategi training CNN (ResNet18) untuk klasifikasi ras kucing & anjing: **scratch vs feature extraction vs fine-tuning**, plus efek **augmentasi**.

## Dataset
- **Oxford-IIIT Pet** (subset 10 kelas: 5 kucing + 5 anjing)
- Split: Train 1400 / Val 300 / Test 300 (stratified, seed=42)

## Model & Setup
- Backbone: **ResNet18** (ImageNet pretrained untuk B2–B4)
- Loss: `CrossEntropyLoss` | Optimizer: `AdamW` (weight decay 1e-2)
- Early stopping: `patience=4`, `max_epochs=30`
- Augmentasi (B1–B3): RandomResizedCrop(224), HFlip, Rotation 15°, ColorJitter
- Device: Colab T4 GPU | Seed: 42

## Konfigurasi Eksperimen
| Kode | Mode | Aug | Trainable Params |
|------|------|-----|------------------|
| B1 | Scratch | ✔ | 11.18M (100%) |
| B2 | Feature Extraction | ✔ | 5.13K (0.05%) |
| B3 | Fine-Tuning (layer4 + fc) | ✔ | 8.40M (75%) |
| B4 | Feature Extraction | ✘ | 5.13K (0.05%) |

## Hasil Test Set
| Exp | Test Acc | Macro-F1 | Gap (train–val) | Durasi |
|-----|----------|----------|-----------------|--------|
| B1_scratch | 0.6300 | 0.6238 | +0.0998 | 7.4 mnt |
| B2_feature_extraction | 0.9600 | 0.9599 | −0.0007 | 7.9 mnt |
| B3_fine_tuning | 0.9633 | 0.9635 | +0.0314 | 2.6 mnt |
| B4_no_aug | 0.9633 | 0.9634 | +0.0167 | 5.3 mnt |

## Kesimpulan
1. **Transfer learning >>> scratch** pada dataset kecil (1400 sampel).
2. **Fine-tuning** paling efisien waktu & akurasi tertinggi (0.9633).
3. **Feature extraction** sudah sangat kuat karena backbone ImageNet; augmentasi memberi efek kecil (+0.01 val acc, gap lebih sehat).
4. **Overfitting** terlihat jelas pada B1 (gap ~0.10); B2–B4 gap < 0.04.
5. **Failure analysis**: 12/300 salah, dominan confusion antar ras visual mirip (Maine Coon ↔ Persian ↔ Bengal, Boxer ↔ Pug).

## Cara Jalankan
Buka notebook di **Google Colab (runtime GPU T4)** → Run All.
