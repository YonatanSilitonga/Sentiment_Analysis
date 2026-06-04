import os
import sys
import pandas as pd
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DIR = PROJECT_ROOT / "Merged_Excel"
OUTPUT_FILE = MERGED_DIR / "Test_combined_preprocessed.xlsx"

# Add project root to sys.path to import local modules
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from sentiment_model.preprocessing import IndonesianTextPreprocessor
except ImportError:
    print("Error: Could not import IndonesianTextPreprocessor. Make sure you are in the project root.")
    sys.exit(1)

def preprocess_test_files():
    print("=" * 60)
    print("MEMULAI PENGGABUNGAN & PREPROCESSING DATA TEST1 - TEST8...")
    print("=" * 60)

    test_files = []
    # Cari file Test1.xlsx sampai Test8.xlsx
    for i in range(1, 9):
        file_path = MERGED_DIR / f"Test{i}.xlsx"
        if file_path.exists():
            test_files.append(file_path)
        else:
            print(f"[-] Peringatan: File {file_path.name} tidak ditemukan di folder Merged_Excel.")

    if not test_files:
        print("[-] Error: Tidak ada file Test1.xlsx - Test8.xlsx yang ditemukan.")
        return

    print(f"[+] Ditemukan {len(test_files)} file untuk digabungkan.")
    
    all_dfs = []
    for file_path in test_files:
        print(f"  [+] Membaca {file_path.name}...")
        try:
            df = pd.read_excel(file_path)
            # Pastikan kolom ulasan ada
            if "ulasan" not in df.columns:
                print(f"  [-] Kolom 'ulasan' tidak ditemukan di {file_path.name}, dilewati.")
                continue
            
            df = df.copy()
            df["sumber_file"] = file_path.name
            all_dfs.append(df)
        except Exception as e:
            print(f"  [-] Gagal membaca {file_path.name}: {e}")

    if not all_dfs:
        print("[-] Error: Tidak ada data yang berhasil dibaca.")
        return

    # Gabungkan semua data
    print("[+] Menggabungkan semua data...")
    merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    initial_rows = len(merged_df)
    print(f"  [+] Total data gabungan awal: {initial_rows} baris.")

    # Bersihkan ulasan kosong
    merged_df["ulasan"] = merged_df["ulasan"].fillna("").astype(str).str.strip()
    merged_df = merged_df[merged_df["ulasan"] != ""].reset_index(drop=True)
    clean_rows = len(merged_df)
    if initial_rows != clean_rows:
        print(f"  [+] Menghapus {initial_rows - clean_rows} baris ulasan kosong.")

    # Preprocessing ulasan
    print("[+] Memulai proses preprocessing teks dengan Sastrawi & Normalisasi Slang...")
    preprocessor = IndonesianTextPreprocessor()
    
    # Hitung progress
    total_ulasan = len(merged_df)
    preprocessed_texts = []
    
    for idx, ulasan in enumerate(merged_df["ulasan"]):
        try:
            preprocessed_texts.append(preprocessor.preprocess_text(ulasan))
        except Exception as e:
            preprocessed_texts.append("")
            print(f"  [-] Gagal memproses ulasan baris ke-{idx+1}: {e}")
        
        # Cetak progress setiap 50 ulasan
        if (idx + 1) % 50 == 0 or (idx + 1) == total_ulasan:
            print(f"  [~] Diproses {idx + 1}/{total_ulasan} ulasan...")

    merged_df["ulasan_preprocessed"] = preprocessed_texts

    # Filter out baris yang setelah dipreprocessing menjadi kosong
    merged_df = merged_df[merged_df["ulasan_preprocessed"].str.strip() != ""].reset_index(drop=True)
    final_rows = len(merged_df)
    print(f"  [+] Menghapus {clean_rows - final_rows} baris yang menjadi kosong setelah preprocessing.")

    # Simpan hasil ke Excel
    print(f"[+] Menyimpan hasil ke {OUTPUT_FILE.name}...")
    try:
        merged_df.to_excel(OUTPUT_FILE, index=False)
        print("=" * 60)
        print("PROSES PREPROCESSING & PENGGABUNGAN SELESAI")
        print(f"Jumlah file asal : {len(test_files)}")
        print(f"Jumlah baris     : {final_rows}")
        print(f"File output      : {OUTPUT_FILE.resolve()}")
        print("=" * 60)
    except Exception as e:
        print(f"[-] Gagal menyimpan file: {e}")

if __name__ == "__main__":
    preprocess_test_files()
