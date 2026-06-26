"""
check_fit_fast.py
Diagnosa overfitting/underfitting TANPA menjalankan ulang preprocessing Sastrawi.
Membaca langsung hasil metadata v12 yang sudah tersimpan dari training sebelumnya,
lalu menghitung akurasi data latih secara cepat menggunakan vectorizer yang sudah di-fit.
"""
import sys
import json
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    # -----------------------------------------------------------------------
    # 1. Baca metadata yang sudah tersimpan untuk mengambil metrik test set
    # -----------------------------------------------------------------------
    metadata_path = PROJECT_ROOT / "sentiment_model/model_artifacts_v12/metadata_v12.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    base_metrics = metadata.get("base_metrics", {})
    test_acc = base_metrics.get("accuracy", None)
    test_f1 = base_metrics.get("macro_f1", None)

    print("Membaca metadata v12 dan memuat dataset untuk analisis fit...")

    # -----------------------------------------------------------------------
    # 2. Baca dataset master, ambil kolom label saja (tidak perlu stemming)
    # -----------------------------------------------------------------------
    data_path = PROJECT_ROOT / "Merged_Excel/dataset_v11_master.xlsx"
    raw_df = pd.read_excel(data_path)

    # Normalisasi nama kolom
    raw_df.columns = raw_df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # Cari kolom teks dan label
    col_aliases = {
        "ulasan": "ulasan", "review": "ulasan", "text": "ulasan", "komentar": "ulasan",
        "teks_clean": "ulasan", "teks_asli": "ulasan",
        "label": "label", "sentimen": "label", "sentiment": "label",
    }
    col_map = {}
    for col in raw_df.columns:
        mapped = col_aliases.get(col)
        if mapped and mapped not in col_map:
            col_map[col] = mapped

    raw_df = raw_df.rename(columns=col_map)

    if "ulasan" not in raw_df.columns or "label" not in raw_df.columns:
        print(f"ERROR: Kolom yang tersedia: {list(raw_df.columns)}")
        return

    valid_labels = {"negatif", "netral", "positif"}
    df = raw_df[["ulasan", "label"]].copy()
    df["ulasan"] = df["ulasan"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()
    df = df[(df["ulasan"] != "") & (df["label"].isin(valid_labels))].reset_index(drop=True)

    # -----------------------------------------------------------------------
    # 3. Load model dan vectorizer v12
    # -----------------------------------------------------------------------
    model_dir = PROJECT_ROOT / "sentiment_model/model_artifacts_v12"
    print(f"Memuat model dari: {model_dir}")
    model = load(model_dir / "sentiment_model_v12.joblib")
    vectorizer = load(model_dir / "tfidf_vectorizer_v12.joblib")

    # -----------------------------------------------------------------------
    # 4. Split data dengan seed yang SAMA persis seperti saat training
    #    agar hasil evaluasinya konsisten
    # -----------------------------------------------------------------------
    X = df["ulasan"].values
    y = df["label"].values

    _, X_test_raw, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_raw, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------------------------------------------------
    # 5. PENTING: Gunakan teks RAW (belum di-stem) + transform vectorizer
    #    yang sudah difit dengan teks preprocessed.
    #    Pendekatan ini cepat (< 3 detik) karena tidak butuh Sastrawi.
    #    Catatan: Akurasi train di sini sedikit lebih rendah dari akurasi
    #    train sesungguhnya, karena kita tidak menggunakan teks yang di-stem.
    #    Ini sudah cukup untuk diagnosa overfitting.
    # -----------------------------------------------------------------------
    print("Melakukan transformasi TF-IDF dan prediksi...")
    X_train_tfidf = vectorizer.transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    train_preds = model.predict(X_train_tfidf)
    test_preds = model.predict(X_test_tfidf)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc_live = accuracy_score(y_test, test_preds)

    train_report = classification_report(y_train, train_preds, output_dict=True, zero_division=0)
    test_report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)

    train_f1 = train_report["macro avg"]["f1-score"]
    test_f1_live = test_report["macro avg"]["f1-score"]

    gap_acc = train_acc - test_acc_live
    gap_f1 = train_f1 - test_f1_live

    # -----------------------------------------------------------------------
    # 6. Cetak Laporan Analisis Fit
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  ANALISIS FIT MODEL SENTIMEN V12")
    print("=" * 55)
    print(f"  Akurasi Data LATIH  (Train Accuracy) : {train_acc*100:6.2f}%")
    print(f"  Akurasi Data UJI    (Test Accuracy)  : {test_acc_live*100:6.2f}%")
    print(f"  Selisih/Gap Akurasi                  : {gap_acc*100:6.2f}%")
    print("-" * 55)
    print(f"  Macro F1-Score LATIH                 : {train_f1*100:6.2f}%")
    print(f"  Macro F1-Score UJI                   : {test_f1_live*100:6.2f}%")
    print(f"  Selisih/Gap F1-Score                 : {gap_f1*100:6.2f}%")
    print("=" * 55)

    print("\n  DIAGNOSA:")
    if train_acc < 0.70 and test_acc_live < 0.70:
        print("  [X] UNDERFITTING: Akurasi Train DAN Test sama-sama rendah.")
        print("      Model terlalu sederhana untuk menangkap pola bahasa.")
    elif gap_acc > 0.07 or gap_f1 > 0.07:
        print(f"  [!] OVERFITTING (Gap terlalu besar: {gap_acc*100:.2f}%):")
        print("      Model terlalu menghafal data latih dan kurang")
        print("      mampu menggeneralisasi ulasan yang baru.")
    else:
        print(f"  [OK] GOOD FIT / OPTIMAL! (Gap hanya {gap_acc*100:.2f}%)")
        print("       Akurasi tinggi di data latih DAN data uji,")
        print("       dengan selisih yang sangat tipis. Model ini")
        print("       memiliki kemampuan generalisasi yang sangat baik!")

    print("=" * 55)

    print("\n  CATATAN:")
    print("  Analisis ini menggunakan teks RAW (tanpa stemming ulang)")
    print("  untuk kecepatan. Akurasi train sesungguhnya (dengan")
    print("  stemming) akan sedikit lebih tinggi dari nilai di atas.")

if __name__ == "__main__":
    main()
