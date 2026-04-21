# Sentiment Analysis Service for Smart Tourism Wisata Toba

Layanan analisis sentimen berbasis Python untuk membaca ulasan wisata Danau Toba, mengklasifikasikan sentimen menjadi negative, neutral, atau positive, lalu mengirim hasil ke admin panel aplikasi.

## Ringkasan Proyek

Proyek ini dibangun untuk kebutuhan Smart Tourism Wisata Toba dengan fokus:

- Analisis sentimen ulasan berbahasa Indonesia.
- Penanganan teks pendek dan teks panjang dengan strategi berbeda.
- Model hybrid: kombinasi machine learning dan aturan kata kunci domain wisata.
- Integrasi API yang mudah dikonsumsi oleh admin panel (Laravel/PHP).

## Tujuan

- Membantu admin memantau persepsi pengunjung secara cepat.
- Memberi insight per ulasan dan agregasi sentimen untuk pengambilan keputusan.
- Menyediakan fondasi untuk dashboard monitoring kualitas layanan destinasi wisata.

## Teknologi

- Python 3.x
- Flask + Flask-CORS
- scikit-learn (TF-IDF + Logistic Regression)
- pandas, numpy, joblib
- Sastrawi (stemming dan stopword Bahasa Indonesia)

## Arsitektur Solusi

Alur utama inferensi:

1. Client admin panel mengirim ulasan ke API Python.
2. API melakukan preprocessing teks Bahasa Indonesia.
3. Model ML melakukan prediksi probabilitas untuk 3 kelas.
4. Hybrid rules menerapkan aturan domain (misalnya sinyal negatif/kontras) untuk koreksi keputusan model.
5. Jika teks panjang/bercampur sentimen, modul long-text melakukan segmentasi dan agregasi skor.
6. API mengembalikan label, confidence, skor per kelas, alasan keputusan, dan metadata pendukung.

## Komponen Inti

- `app.py`
  - Entry point Flask API.
  - Endpoint health check, prediksi single, batch, keyword summary, dan stats.

- `sentiment_model/sentiment_model.py`
  - Service wrapper untuk inferensi.
  - Validasi input/output, mapping label internal-eksternal, summary batch.

- `sentiment_model/preprocessing.py`
  - Normalisasi teks Indonesia, tokenisasi, stopword filtering, stemming.

- `sentiment_model/hybrid.py`
  - Rule engine berbasis sinyal negatif, positif, dan kontras.
  - Menjaga performa pada frase domain wisata yang sering ambigu.

- `sentiment_model/long_text.py`
  - Segmentasi teks panjang dan agregasi prediksi per segmen.

- `sentiment_model/test_model.py`
  - CLI untuk pengujian model (single/batch/interaktif via terminal).

- `sentiment_model/model_artifacts_v9/`
  - Artifact model aktif: model, vectorizer, metadata, dan konfigurasi hybrid.

## Cara Menjalankan Proyek

### 1. Setup environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Jalankan API

```bash
python app.py
```

Secara default API aktif di:

- Host: `127.0.0.1`
- Port: `5000`

## Environment Variables

- `PY_SENTIMENT_MODEL_DIR` (default: `sentiment_model/model_artifacts_v9`)
- `PY_SENTIMENT_HOST` (default: `127.0.0.1`)
- `PY_SENTIMENT_PORT` (default: `5000`)
- `PY_SENTIMENT_DEBUG` (default: `false`)
- `PY_SENTIMENT_ENABLE_CORS` (default: `true`)
- `PY_SENTIMENT_LONG_TEXT_MODE` (default: `auto`)
- `PY_SENTIMENT_LONG_TEXT_MIN_WORDS` (default: `14`)
- `PY_SENTIMENT_LONG_TEXT_MIN_SEGMENTS` (default: `2`)
- `PY_SENTIMENT_CLOSE_MARGIN_NEUTRAL` (default: `0.06`)

Contoh set environment di Windows:

```powershell
$env:PY_SENTIMENT_MODEL_DIR="sentiment_model/model_artifacts_v9"
$env:PY_SENTIMENT_PORT="5000"
python app.py
```

## Endpoint API

### GET `/health`
Memeriksa status service dan model.

### POST `/api/v1/predict`
Prediksi satu ulasan.

Contoh request:

```json
{
  "review_id": 123,
  "text": "Tempatnya indah tapi toiletnya kurang bersih"
}
```

### POST `/api/v1/predict-batch`
Prediksi banyak ulasan sekaligus.

Contoh request:

```json
{
  "reviews": [
    { "id": 1, "text": "Bagus sekali" },
    { "id": 2, "text": "Toiletnya kotor dan bau" }
  ]
}
```

### POST `/api/v1/summary-keywords`
Menghasilkan ringkasan kata kunci berdasarkan hasil sentimen.

### GET `/api/v1/stats`
Menampilkan metadata model dan ringkasan metrik.

## Cara Test Model

### Test satu kalimat

```bash
python -m sentiment_model.test_model --model-dir sentiment_model/model_artifacts_v9 --text "aksesnya bagus dan pemandangannya indah"
```

### Test batch dari file

```bash
python -m sentiment_model.test_model --model-dir sentiment_model/model_artifacts_v9 --input-file Merged_Excel/dataset_labeled_combined.xlsx --output-file Merged_Excel/hasil_prediksi.xlsx
```

### Mode interaktif terminal

```bash
python -m sentiment_model.test_model --model-dir sentiment_model/model_artifacts_v9
```

## Implementasi di Admin Panel Aplikasi Wisata Toba

Skema integrasi umum pada admin panel:

1. Admin membuka halaman manajemen ulasan.
2. Aplikasi mengirim komentar/review ke endpoint `/api/v1/predict` atau `/api/v1/predict-batch`.
3. Response disimpan di database aplikasi sebagai:
   - sentiment label
   - confidence
   - score per kelas
   - reason
   - processed text
4. Halaman admin menampilkan badge sentimen, statistik agregat, dan tren sentimen.

Contoh request dari Laravel:

```php
$response = Http::timeout(5)->post(env('PY_SENTIMENT_URL') . '/api/v1/predict', [
    'review_id' => $review->id,
    'text' => $review->comment,
]);

$result = $response->json();
```

Contoh field yang biasanya dipakai di admin panel:

- `data.label`
- `data.confidence`
- `data.scores.negative`
- `data.scores.neutral`
- `data.scores.positive`
- `data.reason`
- `data.long_text_used`

## Alur Data ke Dashboard Admin

- Input: ulasan wisata dari pengguna aplikasi.
- Proses: inferensi sentimen oleh service Python.
- Output: sentimen per ulasan + ringkasan keyword.
- Visualisasi admin:
  - Distribusi sentimen per destinasi.
  - Daftar ulasan negatif prioritas tindak lanjut.
  - Tren kualitas layanan dari waktu ke waktu.

## Struktur Folder Singkat

```text
Sentiment_Analysis/
├── app.py
├── requirements.txt
├── datasets/
├── docs/
├── experiments/
├── Merged_Excel/
├── sentiment_model/
│   ├── preprocessing.py
│   ├── hybrid.py
│   ├── long_text.py
│   ├── sentiment_model.py
│   ├── test_model.py
│   ├── training/
│   └── model_artifacts_v9/
└── notebooks/
```

## Catatan

- Model aktif default saat ini adalah v9 (`model_artifacts_v9`).
- Script di folder `experiments/` bersifat analisis/diagnostik tambahan, bukan jalur utama production.
- Untuk deployment production, jalankan service Python sebagai process terpisah dan hubungkan URL service melalui environment aplikasi admin.

## Lisensi

Silakan sesuaikan lisensi proyek sesuai kebijakan tim/instansi sebelum publikasi.
