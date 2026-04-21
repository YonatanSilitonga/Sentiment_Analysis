#!/usr/bin/env python
"""Analisis mengapa POSITIF detection lemah di V4"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import json

print("="*80)
print("ANALYSIS: Mengapa POSITIF Detection Lemah?")
print("="*80)

# Load dataset
df_combined = pd.read_excel("Merged_Excel/dataset_labeled_combined.xlsx")
df_original = pd.read_excel("Merged_Excel/dataset_labeled.xlsx")

print("\n📊 DATASET DISTRIBUTION COMPARISON:")
print("-"*80)

# Original
print("\n✅ ORIGINAL DATASET (V1, V2, V3):")
print(f"   Total rows: {len(df_original)}")
dist_orig = df_original['label'].value_counts(normalize=True) * 100
for label, pct in dist_orig.items():
    count = (df_original['label'] == label).sum()
    print(f"   - {label:8s}: {count:3d} rows ({pct:.1f}%)")

# Combined
print("\n⚠️  COMBINED DATASET (V4):")
print(f"   Total rows: {len(df_combined)}")
dist_comb = df_combined['label'].value_counts(normalize=True) * 100
for label, pct in dist_comb.items():
    count = (df_combined['label'] == label).sum()
    print(f"   - {label:8s}: {count:3d} rows ({pct:.1f}%)")

# Delta
print("\n📉 PERUBAHAN (Original → Combined):")
for label in ['negatif', 'positif', 'netral']:
    orig_pct = (df_original['label'] == label).sum() / len(df_original) * 100
    comb_pct = (df_combined['label'] == label).sum() / len(df_combined) * 100
    delta = comb_pct - orig_pct
    symbol = "📈" if delta > 0 else "📉"
    print(f"   {symbol} {label:8s}: {orig_pct:.1f}% → {comb_pct:.1f}% ({delta:+.1f}%)")

print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS:")
print("="*80)

print("""
MASALAH UTAMA: Dataset baru (Toba) "mencemar" keseimbangan data

Breakdowns:
- Original:    53% positif ✅ (Good balance)
- New (Toba):   12.5% positif ❌ (Mostly negative)
- Combined:     28% positif ❌ (Dominated by negative Toba data)

AKIBAT:
1. Model hanya lihat 188 positive examples (vs 345 negative)
2. Model underfit untuk positive pattern recognition
3. Saat test "bersih dan wangi", model tidak yakin (40.6% vs 33.1%)
4. Model default ke NEGATIF

STATISTIK ALARM:
- Training set: hanya ~150 positive examples (80% dari 188)
- Ini VERY SMALL untuk TF-IDF + LogReg model
- Bandingkan dengan: 276 negative examples (81% dari 345)
- Ratio: 1.8x lebih banyak negative training data!

KESIMPULAN:
Model V4 dioptimasi untuk NEGATIVE detection (88.41% recall)
tapi SACRIFICED positive accuracy karena data imbalance!
""")

# Load metadata untuk melihat metrics per class
print("\n" + "="*80)
print("METRICS V4 - BREAKDOWN BY CLASS:")
print("="*80)

meta_v4 = json.loads(Path("sentiment_model/model_artifacts_v4/metadata_v4.json").read_text(encoding='utf-8'))

if 'base_metrics' in meta_v4 and 'classification_report' in meta_v4['base_metrics']:
    report = meta_v4['base_metrics']['classification_report']
    print("\nDetailed Classification Report:")
    for label in ['negatif', 'netral', 'positif']:
        if label in report:
            m = report[label]
            print(f"\n{label.upper()}:")
            print(f"  Precision: {m['precision']:.4f} (dari predicted {label}, berapa benar)")
            print(f"  Recall:    {m['recall']:.4f} (dari actual {label}, berapa detected)")
            print(f"  F1-score:  {m['f1-score']:.4f}")
            print(f"  Support:   {int(m['support'])} examples")

print("\n" + "="*80)
print("SOLUSI UNTUK V5:")
print("="*80)

print("""
OPSI 1: REBALANCE DATASET
- Hapus beberapa negative examples (undersampling)
- Target: 40% negatif, 30% positif, 30% netral
- Hasil: Lebih seimbang training, lebih baik untuk positif
- Risk: Kehilangan negative diversity

OPSI 2: AUGMENT POSITIVE DATA
- Cari lebih banyak positive reviews dataset
- Pakai data augmentation (sinonim, paraphrasing)
- Target: 250-300 positive examples
- Result: Model lebih familiar dengan positive patterns
- Best approach! ✅

OPSI 3: CLASS WEIGHT ADJUSTMENT
- Set class_weight = {
    'negatif': 0.8,
    'positif': 1.5,
    'netral': 1.0
  }
- Hasilnya: Model lebih fokus ke positive minority class
- Mudah, tidak perlu data baru ✅

OPSI 4: ENSEMBLE APPROACH
- Buat 2 model terpisah: negative-vs-rest dan positive-vs-rest
- Combine predictions dengan voting
- Lebih robust ✅

REKOMENDASI: Opsi 2 + Opsi 3 (best results)
- Augment positif data ke 250-300 rows
- Adjust class weights untuk balanced loss
- Retrain → V5 hybrid dengan improved positive detection
""")

print("\n" + "="*80)
