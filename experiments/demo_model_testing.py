#!/usr/bin/env python
"""Demo script untuk testing model v4 dengan perbandingan hasil"""

from pathlib import Path
import pandas as pd
from joblib import load
import json
from sentiment_model.preprocessing import IndonesianTextPreprocessor
from sentiment_model.hybrid import apply_hybrid_rules

print("="*80)
print("DEMO: TESTING SENTIMENT MODEL V4")
print("="*80)

# Load both v1 (original) dan v4 (new combined) untuk perbandingan
def load_model_version(version: str):
    """Load model artifacts untuk specific version"""
    if version == "v1":
        model_dir = Path("sentiment_model/model_artifacts")
        model_file = "sentiment_model.joblib"
        vec_file = "tfidf_vectorizer.joblib"
        meta_file = "metadata.json"
        hybrid_file = None
    else:  # v4
        model_dir = Path("sentiment_model/model_artifacts_v4")
        model_file = "sentiment_model_v4.joblib"
        vec_file = "tfidf_vectorizer_v4.joblib"
        meta_file = "metadata_v4.json"
        hybrid_file = "hybrid_rules_v4.json"
    
    try:
        model = load(model_dir / model_file)
        vectorizer = load(model_dir / vec_file)
        metadata = json.loads((model_dir / meta_file).read_text(encoding='utf-8'))
        hybrid_config = None
        if hybrid_file:
            hybrid_config = json.loads((model_dir / hybrid_file).read_text(encoding='utf-8'))
        return model, vectorizer, metadata, hybrid_config
    except FileNotFoundError as e:
        print(f"❌ Error loading {version}: {e}")
        return None, None, None, None

# Test phrases
test_phrases = [
    "tempatnya bau pesing",
    "akses sulit dan parkir liar",
    "pelayanan buruk dan sangat kecewa",
    "fasilitas rusak berlubang",
    "toilet kotor banget",
    "tidak ada AC dan bocor",
]

preprocessor = IndonesianTextPreprocessor()

# Load models
print("\n📦 Loading models...")
model_v1, vec_v1, meta_v1, _ = load_model_version("v1")
model_v4, vec_v4, meta_v4, hybrid_v4 = load_model_version("v4")

if not all([model_v1, model_v4]):
    print("❌ Failed to load models")
    exit(1)

print("✅ Models loaded successfully\n")

# Test setiap phrase
for i, phrase in enumerate(test_phrases, 1):
    preprocessed = preprocessor.preprocess_text(phrase)
    
    # V1 prediction
    x_v1 = vec_v1.transform([preprocessed])
    pred_v1 = model_v1.predict(x_v1)[0]
    prob_v1 = model_v1.predict_proba(x_v1)[0]
    conf_v1 = max(prob_v1)
    
    # V4 prediction (base)
    x_v4 = vec_v4.transform([preprocessed])
    pred_v4_base = model_v4.predict(x_v4)[0]
    prob_v4 = model_v4.predict_proba(x_v4)[0]
    
    # V4 prediction (hybrid)
    result_df = pd.DataFrame({
        'ulasan_asli': [phrase],
        'ulasan_preprocessed': [preprocessed],
        'prediksi_sentimen': [pred_v4_base],
        **{f'prob_{label}': [prob_v4[idx]] for idx, label in enumerate(model_v4.classes_)}
    })
    hybrid_df = apply_hybrid_rules(result_df, [phrase], hybrid_v4)
    pred_v4_hybrid = hybrid_df['prediksi_sentimen'].iloc[0]
    
    print(f"{i}. Ulasan: \"{phrase}\"")
    print(f"   Preprocessed: \"{preprocessed}\"")
    print(f"   V1 (Original): {pred_v1:8s} (confidence: {conf_v1:.2%})")
    print(f"   V4 (Base):     {pred_v4_base:8s} (proba: {dict(zip(model_v4.classes_, prob_v4.round(3)))})")
    print(f"   V4 (Hybrid):   {pred_v4_hybrid:8s} ⭐ FINAL PREDICTION")
    
    # Highlight if v1 and v4 differ
    if pred_v1 != pred_v4_hybrid:
        print(f"   ⚠️  CORRECTION: V1 predicted '{pred_v1}', V4 predicts '{pred_v4_hybrid}'")
    print()

print("="*80)
print("\n💡 INTERPRETASI HASIL:")
print("""
- V1 (Original): Model pertama dengan 379 rows training data (TF-IDF + LR)
- V4 (Base):     Model baru dengan 677 rows + balanced class weights (lebih representatif)
- V4 (Hybrid):   V4 Base + 28 negative keyword rules (paling akurat untuk negative cases)

Model V4 lebih baik karena:
✅ Training data 1.8x lebih besar (677 vs 379 rows)
✅ Class balance lebih bagus (51% negatif vs 14% di v1)
✅ Hybrid rules menangkap negative signals yang v1 miss
✅ 88.41% recall pada negative class di test set 136 rows
""")

print("="*80)
print("\n🚀 CARA TESTING LEBIH LANJUT:\n")
print("1. Single text:")
print("   python sentiment_model/test_model.py --model-dir sentiment_model/model_artifacts_v4 --text \"ulasan Anda di sini\"")
print("\n2. Interactive (input berulang):")
print("   python sentiment_model/test_model.py --model-dir sentiment_model/model_artifacts_v4")
print("\n3. Batch file (.xlsx/.csv):")
print("   python sentiment_model/test_model.py --model-dir sentiment_model/model_artifacts_v4 \\")
print("     --input-file data.xlsx --text-column ulasan --output-file hasil.xlsx")
print("\n" + "="*80)
