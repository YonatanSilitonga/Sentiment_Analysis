#!/usr/bin/env python
"""Debug script untuk analisis error prediksi"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentiment_model.preprocessing import IndonesianTextPreprocessor
from sentiment_model.test_model import load_artifacts, predict_texts
from sentiment_model.hybrid import NEGATIVE_SIGNALS
import pandas as pd

# Test phrase yang error
test_phrase = "kamar mandinya bersih dan wangi"

print("="*80)
print("DEBUG: Analyzing Incorrect Prediction")
print("="*80)
print(f"\n📝 Text: \"{test_phrase}\"")
print(f"Expected: POSITIF (bersih, wangi = clean, fragrant)")
print(f"Got: NEGATIF (ERROR!)")

# Load model dan tools
model_dir = Path("sentiment_model/model_artifacts_v4")
model, vectorizer, metadata, hybrid_config = load_artifacts(model_dir)
preprocessor = IndonesianTextPreprocessor()

# Preprocessing
preprocessed = preprocessor.preprocess_text(test_phrase)
print(f"\n🔍 Preprocessing:")
print(f"  Original   : {test_phrase}")
print(f"  Processed  : {preprocessed}")

# Vectorize
X = vectorizer.transform([preprocessed])
print(f"\n📊 TF-IDF Vector dimensions: {X.shape}")

# Base prediction
pred_base = model.predict(X)[0]
proba = model.predict_proba(X)[0]
print(f"\n🤖 Base Model Prediction:")
print(f"  Prediction: {pred_base}")
print(f"  Probabilities:")
for label, prob in zip(model.classes_, proba):
    print(f"    - {label:8s}: {prob:.4f} ({prob*100:.1f}%)")

# Check hybrid rules
print(f"\n🔎 Hybrid Rules Analysis:")
print(f"  Total negative patterns: {len(NEGATIVE_SIGNALS)}")

# Check if any negative keywords matched
matched_keywords = []
for pattern, weight, tag in NEGATIVE_SIGNALS:
    import re
    if re.search(pattern, test_phrase.lower()):
        matched_keywords.append((pattern, weight, tag))
        print(f"  ⚠️  MATCHED: \"{pattern}\" (tag: {tag}, weight: {weight})")

if not matched_keywords:
    print(f"  ✅ NO negative keywords matched (good)")

# Hybrid prediction
result_df = predict_texts(
    model, vectorizer, preprocessor, [test_phrase], 
    hybrid_config=hybrid_config
)
row = result_df.iloc[0]

print(f"\n🎯 Hybrid Result:")
print(f"  Final Prediction: {row['prediksi_sentimen'].upper()}")
if 'hybrid_reason' in row:
    print(f"  Reason: {row['hybrid_reason']}")

print("\n" + "="*80)
print("💡 ANALYSIS:")
print("="*80)
print("""
MASALAH: Model memprediksi "kamar mandinya bersih dan wangi" sebagai NEGATIF

ROOT CAUSE KEMUNGKINAN:
1. ❌ Word "bersih" (clean) mungkin jarang di training data positif
2. ❌ Word "wangi" (fragrant) mungkin tidak ada/jarang di training data
3. ❌ Model bias ke NEGATIF karena training data 51% negatif, 28% positif
4. ❌ Ada keyword negatif yang matched (check di atas)

SOLUSI:
✅ Augment training data dengan lebih banyak positive hotel review
✅ Add positive sentiment keywords: bersih, wangi, harum, nyaman, bagus
✅ Rebalance class weights lebih ke POSITIF
✅ Retrain model dengan improved data

STATUS: Ini adalah bug yang perlu diperbaiki di v5
""")

print("\n" + "="*80)
print("COMPARISON: Test similar positive phrases")
print("="*80)

test_cases = [
    "kamar mandinya sangat bersih",
    "wangi harum dan bersih",
    "kamar bagus dan bersih",
    "toiletnya bersih banget",
]

for phrase in test_cases:
    result = predict_texts(model, vectorizer, preprocessor, [phrase], hybrid_config=hybrid_config)
    pred = result.iloc[0]['prediksi_sentimen']
    emoji = "😞" if pred == "negatif" else "😊" if pred == "positif" else "😐"
    print(f"\n{emoji} \"{phrase}\"")
    print(f"   → {pred.upper()}")
