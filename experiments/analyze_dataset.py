import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

def analyze_dataset():
    file_path = Path("Merged_Excel/dataset_v11_master.xlsx")
    if not file_path.exists():
        print(f"[X] File tidak ditemukan: {file_path}")
        return
        
    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path)
    
    print("\n" + "="*50)
    print("DETAIL DATASET MASTER V11")
    print("="*50)
    print(f"Jumlah Baris       : {len(df)}")
    print(f"Jumlah Kolom       : {len(df.columns)}")
    print(f"Daftar Kolom       : {list(df.columns)}")
    
    # 1. Check text and label columns
    text_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower in ['ulasan', 'text', 'review', 'review_text', 'ulasan_asli', 'ulasan_preprocessed', 'sentimen_preprocess']:
            text_col = col
        if col_lower in ['label', 'sentimen', 'sentiment', 'category', 'kelas', 'prediksi_sentimen']:
            label_col = col
            
    if not text_col:
        # Fallback to the first string column that has long average length
        for col in df.columns:
            if df[col].dtype == object:
                sample_len = df[col].astype(str).str.len().mean()
                if sample_len > 15:
                    text_col = col
                    break
    if not label_col:
        # Fallback to column with categorical distribution (few unique values)
        for col in df.columns:
            if col != text_col:
                unique_cnt = df[col].nunique()
                if 2 <= unique_cnt <= 5:
                    label_col = col
                    break
                    
    print(f"Kolom Teks Terdeteksi  : '{text_col}'")
    print(f"Kolom Label Terdeteksi : '{label_col}'")
    
    if not text_col or not label_col:
        print("[X] Kolom teks atau label tidak terdeteksi secara otomatis!")
        return

    # Clean text for grouping
    df['text_clean'] = df[text_col].fillna("").astype(str).str.strip().str.lower()
    
    # 2. Label Distribution
    print("\nDISTRIBUSI KELAS (SENTIMEN):")
    dist = df[label_col].value_counts(dropna=False)
    for label, count in dist.items():
        percentage = (count / len(df)) * 100
        print(f" - {label:<10}: {count:>6} baris ({percentage:.2f}%)")
        
    # 3. Missing values
    missing_text = df[text_col].isna().sum()
    missing_label = df[label_col].isna().sum()
    print(f"\nKELENGKAPAN DATA:")
    print(f" - Missing Teks  : {missing_text} baris")
    print(f" - Missing Label : {missing_label} baris")

    # 4. Duplicate texts with same labels
    duplicates_same_label = df[df.duplicated(subset=['text_clean', label_col], keep=False)]
    unique_dupes = duplicates_same_label['text_clean'].nunique()
    print(f" - Duplikat dengan Label Sama : {len(duplicates_same_label)} baris ({unique_dupes} teks unik)")

    # 5. Duplicate texts with CONFLICTING labels (very critical issue!)
    grouped = df.groupby('text_clean')[label_col].nunique()
    conflicting_texts = grouped[grouped > 1].index.tolist()
    
    print("\nDETEKSI PELABELAN KONFLIK (Teks Sama, Label Berbeda):")
    print(f"Jumlah Teks Konflik: {len(conflicting_texts)}")
    
    if conflicting_texts:
        print("-" * 50)
        # Show first 10 conflicting cases
        for idx, text in enumerate(conflicting_texts[:10]):
            sub_df = df[df['text_clean'] == text]
            labels_found = sub_df[label_col].tolist()
            orig_texts = sub_df[text_col].tolist()
            print(f"{idx+1}. Teks: \"{orig_texts[0]}\"")
            for label, orig in zip(labels_found, orig_texts):
                print(f"   -> Label: [{label}]")
        if len(conflicting_texts) > 10:
            print(f"...dan {len(conflicting_texts) - 10} konflik lainnya.")
    else:
        print("[v] Tidak ditemukan teks duplikat dengan label konflik!")
        
    # 6. Potential mislabeled reviews using simple rule checking
    neg_indicators = ['kotor', 'buruk', 'kecewa', 'mahal', 'rusak', 'bau', 'lambat', 'antre', 'menyesal', 'kapok']
    pos_indicators = ['indah', 'bagus', 'bersih', 'enak', 'ramah', 'puas', 'mantap', 'murah', 'nyaman', 'rekomendasi']
    
    print("\nPENGECEKAN POTENSI SALAH LABEL (SAMPEL):")
    
    # Labeled Positive but containing negative words without positive words
    pos_but_neg = []
    for idx, row in df.iterrows():
        t = str(row[text_col]).lower()
        lbl = str(row[label_col]).lower()
        if lbl in ['positif', 'positive']:
            has_neg = any(w in t for w in neg_indicators)
            has_pos = any(w in t for w in pos_indicators)
            if has_neg and not has_pos:
                pos_but_neg.append((row[text_col], row[label_col]))
                
    print(f" - Labeled POSITIF tapi mengandung indikator NEGATIF tanpa indikator POSITIF: {len(pos_but_neg)} baris")
    if pos_but_neg:
        for text, label in pos_but_neg[:5]:
            print(f"   -> \"{text[:100]}\"")
            
    # Labeled Negative but containing positive words without negative words
    neg_but_pos = []
    for idx, row in df.iterrows():
        t = str(row[text_col]).lower()
        lbl = str(row[label_col]).lower()
        if lbl in ['negatif', 'negative']:
            has_neg = any(w in t for w in neg_indicators)
            has_pos = any(w in t for w in pos_indicators)
            if has_pos and not has_neg:
                neg_but_pos.append((row[text_col], row[label_col]))
                
    print(f" - Labeled NEGATIF tapi mengandung indikator POSITIF tanpa indikator NEGATIF: {len(neg_but_pos)} baris")
    if neg_but_pos:
        for text, label in neg_but_pos[:5]:
            print(f"   -> \"{text[:100]}\"")

if __name__ == "__main__":
    analyze_dataset()
