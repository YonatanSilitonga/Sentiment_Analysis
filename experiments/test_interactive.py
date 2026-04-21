#!/usr/bin/env python
"""Wrapper untuk testing model dengan UI yang lebih user-friendly"""

import sys
from pathlib import Path

try:
    from sentiment_model.preprocessing import IndonesianTextPreprocessor
    from sentiment_model.test_model import load_artifacts, predict_texts
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sentiment_model.preprocessing import IndonesianTextPreprocessor
    from sentiment_model.test_model import load_artifacts, predict_texts


def test_interactive(model_dir: str = "sentiment_model/model_artifacts_v4"):
    """Interactive testing dengan error handling"""
    
    print("="*70)
    print("MODEL SENTIMENT ANALYZER - INTERACTIVE MODE")
    print("="*70)
    
    try:
        # Load model
        print("\n📦 Loading model...", end=" ", flush=True)
        model_path = Path(model_dir)
        
        if not model_path.exists():
            print(f"❌\nModel folder tidak ditemukan: {model_path.resolve()}")
            return
        
        model, vectorizer, metadata, hybrid_config = load_artifacts(model_path)
        print("✅")
        
        # Display info
        if metadata:
            print(f"\n📊 Model Info:")
            print(f"   Version: {metadata.get('version', 'unknown')}")
            print(f"   Dataset: {metadata.get('data_source', 'unknown')}")
        
        if hybrid_config:
            print(f"   Mode: Hybrid (ML + {len([k for k in hybrid_config if 'threshold' in k])} rules)")
        
        preprocessor = IndonesianTextPreprocessor()
        
        print("\n" + "-"*70)
        print("Ketik ulasan untuk analisis (ketik 'exit' untuk keluar)")
        print("-"*70 + "\n")
        
        while True:
            try:
                # Input dengan error handling
                user_input = input("📝 Ulasan: ").strip()
                
                if not user_input:
                    print("   ⚠️  Ulasan kosong. Coba lagi.\n")
                    continue
                
                if user_input.lower() in {"exit", "quit", "keluar", "selesai"}:
                    print("\n👋 Selesai. Terima kasih!")
                    break
                
                # Predict
                try:
                    result_df = predict_texts(
                        model, vectorizer, preprocessor, [user_input], 
                        hybrid_config=hybrid_config
                    )
                    row = result_df.iloc[0]
                    
                    pred = row['prediksi_sentimen']
                    emoji = "😞" if pred == "negatif" else "😊" if pred == "positif" else "😐"
                    
                    print(f"\n   {emoji} Prediksi: {pred.upper()}")
                    
                    # Show probabilities
                    prob_cols = [col for col in result_df.columns if col.startswith("prob_")]
                    if prob_cols:
                        probs = {col.replace("prob_", ""): row[col] for col in prob_cols}
                        prob_str = ", ".join(f"{k}={v:.1%}" for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True))
                        print(f"   📊 Confidence: {prob_str}")
                    
                    # Show hybrid reason
                    if "hybrid_reason" in result_df.columns:
                        reason = row.get("hybrid_reason", "")
                        if reason and reason != "model_only":
                            print(f"   🔍 Hybrid reason: {reason}")
                    
                    print()
                    
                except Exception as e:
                    print(f"   ❌ Error saat prediksi: {e}\n")
                    
            except EOFError:
                print("\n👋 Input ended. Selesai.")
                break
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Selesai.")
                break
            except Exception as e:
                print(f"   ❌ Error: {e}\n")
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive sentiment model tester")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="sentiment_model/model_artifacts_v4",
        help="Path ke model artifacts folder"
    )
    args = parser.parse_args()
    
    test_interactive(model_dir=args.model_dir)
