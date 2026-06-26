import sys
from pathlib import Path
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentiment_model.preprocessing import IndonesianTextPreprocessor, LABEL_ORDER, prepare_labeled_dataframe

def main():
    print("Membaca dataset dan melakukan preprocessing...")
    data_path = PROJECT_ROOT / "Merged_Excel/dataset_v11_master.xlsx"
    raw_df = pd.read_excel(data_path)
    df = prepare_labeled_dataframe(raw_df, LABEL_ORDER)
    
    preprocessor = IndonesianTextPreprocessor()
    df["ulasan_preprocessed"] = df["ulasan"].apply(preprocessor.preprocess_text)
    df = df[df["ulasan_preprocessed"].str.strip() != ""].reset_index(drop=True)
    
    X_text = df["ulasan_preprocessed"]
    X_raw = df["ulasan"]
    y = df["label"]
    
    # Split data persis seperti skrip training
    X_text_train, X_text_test, X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_text,
        X_raw,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
    # Load model artifacts v12
    print("Memuat model dan vectorizer v12...")
    model_dir = PROJECT_ROOT / "sentiment_model/model_artifacts_v12"
    model = load(model_dir / "sentiment_model_v12.joblib")
    vectorizer = load(model_dir / "tfidf_vectorizer_v12.joblib")
    
    # Transform data
    X_train_tfidf = vectorizer.transform(X_text_train)
    X_test_tfidf = vectorizer.transform(X_text_test)
    
    # Prediksi
    train_preds = model.predict(X_train_tfidf)
    test_preds = model.predict(X_test_tfidf)
    
    # Evaluasi metrik
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    train_report = classification_report(y_train, train_preds, output_dict=True)
    test_report = classification_report(y_test, test_preds, output_dict=True)
    
    train_f1 = train_report["macro avg"]["f1-score"]
    test_f1 = test_report["macro avg"]["f1-score"]
    
    print("\n" + "="*50)
    print("HASIL ANALISIS FIT (OVERFITTING VS UNDERFITTING)")
    print("="*50)
    print(f"Akurasi Data Latih (Train Accuracy) : {train_acc*100:.2f}%")
    print(f"Akurasi Data Uji (Test Accuracy)   : {test_acc*100:.2f}%")
    print(f"Selisih Akurasi (Gap Accuracy)     : {(train_acc - test_acc)*100:.2f}%")
    print("-"*50)
    print(f"Macro F1-Score (Train)             : {train_f1*100:.2f}%")
    print(f"Macro F1-Score (Test)              : {test_f1*100:.2f}%")
    print(f"Selisih F1-Score (Gap F1)          : {(train_f1 - test_f1)*100:.2f}%")
    print("="*50)
    
    # Kesimpulan analitis
    gap = train_acc - test_acc
    if train_acc < 0.70 and test_acc < 0.70:
        print("Kesimpulan: Model mengalami UNDERFITTING (akurasi train dan test sama-sama rendah).")
    elif gap > 0.07:
        print(f"Kesimpulan: Model cenderung OVERFITTING (gap akurasi latih dan uji cukup besar: {gap*100:.2f}%).")
    else:
        print(f"Kesimpulan: Model dalam kondisi GOOD FIT / OPTIMAL (gap akurasi hanya {gap*100:.2f}%).")
    print("="*50)

if __name__ == "__main__":
    main()
