# Dokumentasi Data dan Metrik Model Analisis Sentimen (v11 & v12)
## Sistem Smart Tourism Danau Toba Berbasis Analisis Sentimen Hibrida

Dokumen ini menjelaskan secara rinci data yang digunakan, pembagian dataset, penyeimbangan kelas, serta hasil evaluasi metrik kinerja komparatif untuk **Model Analisis Sentimen Versi 11 dan Versi 12 (Terkini)**.

---

### 1. Deskripsi Dataset Utama (v11 & v12)
Kedua model (v11 dan v12) dilatih menggunakan dataset gabungan master yang sama, yang disimpan pada berkas [dataset_v11_master.xlsx](file:///d:/semester-4-IT%20Del/Semester%20VI/UI-UX%20DESIGN/Sentiment_Analysis/Merged_Excel/dataset_v11_master.xlsx). Dataset ini memiliki total **13.027 baris data ulasan** setelah melalui proses prapemrosesan (*preprocessing*).

#### A. Komposisi Sumber Data
Dataset master dibentuk dengan menggabungkan data lokal pariwisata Toba dengan dataset sentimen Bahasa Indonesia eksternal guna memperkaya variasi kosakata dan model linguistik:
1. **`train_preprocess_ori.tsv` (84.4%)**: 11.000 baris ulasan umum berbahasa Indonesia.
2. **`hotel_sentimen_berlabel.xlsx` (6.1%)**: 800 baris ulasan spesifik mengenai akomodasi/hotel.
3. **Data Pariwisata Toba Lokal (9.5%)**: 1.227 baris data gabungan dari berkas base [dataset_labeled_combined_v3.xlsx](file:///d:/semester-4-IT%20Del/Semester%20VI/UI-UX%20DESIGN/Sentiment_Analysis/Merged_Excel/dataset_labeled_combined_v3.xlsx), yang terdiri atas:
   - **Unspecified/Lainnya**: 677 baris
   - **Sintetik (Generasi Manual)**: 350 baris (ulasan pariwisata Toba terarah)
   - **Google Maps**: 132 baris ulasan riil destinasi wisata Toba
   - **TripAdvisor**: 68 baris ulasan riil destinasi wisata Toba

#### B. Distribusi Sentimen Awal (Sebelum Rebalancing)
Sebaran label asli pada dataset 13.027 baris ini bersifat tidak seimbang (*imbalanced*):
* **Positif**: 7.567 baris (58.1%)
* **Negatif**: 3.811 baris (29.2%)
* **Netral**: 1.649 baris (12.7%)

---

### 2. Pembagian Dataset (Dataset Division)
Untuk proses pelatihan dan evaluasi kedua model, dataset dibagi menggunakan metode split terstratifikasi (*stratified split*) dengan proporsi **80% data latih (train)** dan **20% data uji (test)**:
* **Total Data Latih (Train Set)**: **10.421 baris**
* **Total Data Uji (Test Set)**: **2.606 baris**

---

### 3. Penyeimbangan Kelas & Parameter Pelatihan (Class Balancing)
Kedua model menerapkan penyeimbangan data latih dan parameter bobot yang sama untuk mengatasi ketidakseimbangan kelas (*class imbalance*):

#### A. Rebalancing Data Latih
Data latih sebanyak 10.421 baris di-resample agar mencapai rasio target: **34% negatif**, **33% positif**, dan **33% netral**. Detail perubahannya adalah:
* **Negatif**: dari 3.049 baris $\rightarrow$ **3.545 baris** *(Oversampled)*
* **Positif**: dari 6.053 baris $\rightarrow$ **3.438 baris** *(Undersampled)*
* **Netral**: dari 1.319 baris $\rightarrow$ **3.438 baris** *(Oversampled)*

#### B. Parameter Class Weights pada Model
Pada tahap pelatihan algoritma Logistic Regression, digunakan bobot penalti kustom (**`custom_v1`**) untuk memberikan prioritas lebih tinggi pada kelas sentimen negatif:
* `negatif`: **1.05**
* `netral`: **1.00**
* `positif`: **0.95**

---

### 4. Perbandingan Konfigurasi & Kinerja Model (v11 vs v12)

Perbedaan mendasar antara v11 dan v12 terletak pada konfigurasi **Rule Engine (Hybrid Rules)** di atas model Machine Learning dasar:

* **Versi 11 (`v11_expanded_data_slang_lexicon`)**:
  Mengaktifkan aturan *Neutral Override* (`"neutral_override_enabled": true`). Ulasan dengan kata kunci netral atau skor kata kunci yang minim dipaksa menjadi kelas `netral`.
* **Versi 12 (`v12_optimized_hybrid_negation_rules`)**:
  Menonaktifkan aturan *Neutral Override* (`"neutral_override_enabled": false`) dan memperkuat deteksi cakupan negasi (*negation scope*) untuk sentimen negatif.

#### Tabel Perbandingan Metrik Evaluasi (Data Uji: 2.606 baris)

| Metrik Evaluasi | Base Model (v11/v12) | Hybrid Model v11 (Degradasi) | Hybrid Model v12 (Terkini/Optimal) |
| :--- | :---: | :---: | :---: |
| **Akurasi Keseluruhan** | **84.23%** | 73.79% | **83.04%** |
| **Macro F1-Score** | **80.47%** | 69.67% | **78.94%** |
| **Recall Negatif** | 81.63% | 81.23% | **83.86%** *(Tertinggi)* |
| **Recall Netral** | 79.70% | **76.36%** | 72.73% |
| **Recall Positif** | 86.53% | 69.48% | **84.87%** |

> [!IMPORTANT]
> Aturan *neutral override* pada **v11** terbukti terlalu agresif, sehingga menjatuhkan recall kelas positif dari **86.53%** menjadi **69.48%** karena banyak ulasan positif panjang yang terdeteksi salah sebagai netral. Pada **v12**, dengan mematikan aturan ini, kinerja klasifikasi positif berhasil dipulihkan mendekati performa base model (**84.87%**) sekaligus meningkatkan sensitivitas deteksi ulasan negatif ke angka tertinggi (**83.86%**).

---

### 5. Confusion Matrix Komparatif

#### A. Confusion Matrix - Hybrid Model v11 (Degradasi)
```
                  Predicted Negatif   Predicted Netral   Predicted Positif
True Negatif             619                 103                 40          (Total: 762)
True Netral               37                 252                 41          (Total: 330)
True Positif             139                 323                1052          (Total: 1514)
```
* **Kelemahan v11**: Sebanyak **323 ulasan positif** dan **103 ulasan negatif** salah dikelompokkan sebagai netral oleh sistem override.

#### B. Confusion Matrix - Hybrid Model v12 (Terkini)
```
                  Predicted Negatif   Predicted Netral   Predicted Positif
True Negatif             639                 69                 54          (Total: 762)
True Netral               41                240                 49          (Total: 330)
True Positif             174                 55               1285          (Total: 1514)
```
* **Kelebihan v12**: Klasifikasi benar pada ulasan positif meningkat pesat dari 1.052 menjadi **1.285** ulasan, sementara klasifikasi benar pada ulasan negatif meningkat menjadi **639** ulasan.

---

### 6. Berkas Referensi Terkait
* Preprocessing Pipeline: [preprocessing.py](file:///d:/semester-4-IT%20Del/Semester%20VI/UI-UX%20DESIGN/Sentiment_Analysis/sentiment_model/preprocessing.py)
* Aturan Hibrida Terkini: [hybrid.py](file:///d:/semester-4-IT%20Del/Semester%20VI/UI-UX%20DESIGN/Sentiment_Analysis/sentiment_model/hybrid.py)
* Script Pelatihan v11: [train_hybrid_v11.py](file:///d:/semester-4-IT%20Del/Semester%20VI/UI-UX%20DESIGN/Sentiment_Analysis/sentiment_model/training/train_hybrid_v11.py)
* Script Pelatihan v12: [train_hybrid_v12.py](file:///d:/semester-4-IT%20Del/Semester%20VI/UI-UX%20DESIGN/Sentiment_Analysis/sentiment_model/training/train_hybrid_v12.py)
* Artefak Metadata v11: [metadata_v11.json](file:///d:/semester-4-IT%20Del/Semester%20VI/UI-UX%20DESIGN/Sentiment_Analysis/sentiment_model/model_artifacts_v11/metadata_v11.json)
* Artefak Metadata v12: [metadata_v12.json](file:///d:/semester-4-IT%20Del/Semester%20VI/UI-UX%20DESIGN/Sentiment_Analysis/sentiment_model/model_artifacts_v12/metadata_v12.json)
