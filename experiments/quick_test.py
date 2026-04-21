#!/usr/bin/env python
"""Simple testing script - minimal dependencies"""

import sys
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    try:
        from sentiment_model.preprocessing import IndonesianTextPreprocessor
        from sentiment_model.test_model import load_artifacts, predict_texts
    except ImportError as e:
        print(f"❌ Import error: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("="*70)
    print("🎯 SENTIMENT ANALYZER - V4 MODEL")
    print("="*70)
    
    # Load model
    try:
        model_dir = Path("sentiment_model/model_artifacts_v4")
        print(f"\n📦 Loading from: {model_dir.resolve()}")
        
        model, vectorizer, metadata, hybrid_config = load_artifacts(model_dir)
        print("✅ Model loaded successfully\n")
        
        preprocessor = IndonesianTextPreprocessor()
        
        # Test examples
        examples = [
            "bau pesing",
            "toilet kotor",
            "akses sulit parkir liar",
            "pelayanan buruk",
            "fasilitas rusak berlubang",
        ]
        
        print("Test Examples:")
        print("-" * 70)
        
        for text in examples:
            result_df = predict_texts(
                model, vectorizer, preprocessor, [text], 
                hybrid_config=hybrid_config
            )
            row = result_df.iloc[0]
            pred = row['prediksi_sentimen']
            
            emoji = "😞" if pred == "negatif" else "😊" if pred == "positif" else "😐"
            print(f"\n{emoji} \"{text}\"")
            print(f"   Prediksi: {pred.upper()}")
            
            if "hybrid_reason" in row and row['hybrid_reason']:
                print(f"   Reason: {row['hybrid_reason']}")
        
        print("\n" + "="*70)
        print("✅ Testing complete!")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
