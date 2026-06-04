import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentiment_model.test_model import load_artifacts
from sentiment_model.preprocessing import IndonesianTextPreprocessor

def main():
    print("Membangun visualisasi V11...")
    
    # 1. Load Artifacts
    MODEL_DIR = PROJECT_ROOT / 'sentiment_model' / 'model_artifacts_v11'
    model, vectorizer, metadata, _ = load_artifacts(MODEL_DIR)
    
    # 2. Plot Confusion Matrix
    hybrid_metrics = metadata.get('hybrid_metrics', {})
    cm = np.array(hybrid_metrics.get('confusion_matrix', [[0,0,0],[0,0,0],[0,0,0]]))
    labels = metadata.get('label_order', ['negatif', 'netral', 'positif'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - Hybrid Model V11 (13K Data)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    output_dir = PROJECT_ROOT / 'reports' / 'v11_plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm_path = output_dir / 'confusion_matrix_v11.png'
    plt.savefig(cm_path)
    print(f"Confusion Matrix disimpan di: {cm_path}")
    plt.close()

    # 3. Generate WordClouds
    print("Menghasilkan Word Clouds (ini mungkin memakan waktu beberapa detik)...")
    DATA_PATH = PROJECT_ROOT / 'Merged_Excel' / 'dataset_v11_master.xlsx'
    df = pd.read_excel(DATA_PATH)
    
    preprocessor = IndonesianTextPreprocessor()
    
    for sentiment, color in [('positif', 'Greens'), ('negatif', 'Reds'), ('netral', 'Blues')]:
        text = " ".join(df[df['label'] == sentiment]['ulasan'].astype(str).apply(lambda x: preprocessor.preprocess_text(x)))
        wc = WordCloud(width=800, height=400, background_color='white', colormap=color, max_words=50).generate(text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f"Word Cloud - Sentimen {sentiment.upper()}", fontsize=20)
        plt.axis('off')
        
        wc_path = output_dir / f'wordcloud_{sentiment}_v11.png'
        plt.savefig(wc_path)
        print(f"WordCloud {sentiment} disimpan di: {wc_path}")
        plt.close()

    # 4. Top Features
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    for i, (label, color) in enumerate(zip(labels, ['magma', 'viridis', 'plasma'])):
        top_indices = np.argsort(coefs[i])[-20:]
        top_features = [feature_names[idx] for idx in top_indices]
        top_coefs = [coefs[i][idx] for idx in top_indices]
        
        sns.barplot(x=top_coefs, y=top_features, ax=axes[i], palette=color)
        axes[i].set_title(f"Top 20 Features for {label.upper()}")

    feat_path = output_dir / 'top_features_v11.png'
    plt.savefig(feat_path)
    print(f"Feature Importance disimpan di: {feat_path}")
    plt.close()

    print("\nEvaluasi Selesai! Semua gambar tersedia di folder 'reports/v11_plots/'")

if __name__ == "__main__":
    main()
