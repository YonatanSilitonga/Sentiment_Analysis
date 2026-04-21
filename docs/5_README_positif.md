# README — Dataset Sentimen Positif-Netral Wisata Toba

## 1. Gambaran Umum

Dataset ini merupakan **paket kedua** dari rangkaian dataset sentimen destinasi wisata
berbahasa Indonesia, difokuskan untuk memperbaiki performa model klasifikasi kelas
**POSITIF dan NETRAL**. Seluruh data bersifat sintetik/augmentasi.

- **Total data**: 350 baris
- **Positif**: 250 baris (71.4%)
- **Netral**: 100 baris (28.6%)
- **Negatif**: 0 baris (tidak dimasukkan di paket ini)

---

## 2. Metode Pengumpulan/Generasi

- **Pendekatan**: Generasi manual berbasis konteks domain wisata Danau Toba
- **Gaya bahasa**: Bervariasi — formal (15%), semi-formal (75%), slang/informal (≤10%)
- **Panjang ulasan**: Pendek (< 80 karakter), sedang (80–150 karakter), panjang (> 150 karakter)
- **Distribusi topik**: Mengikuti target distribusi yang ditetapkan (kebersihan+pelayanan ≥ 36%)
- **Domain khusus**: Konten banyak merujuk lokasi spesifik Danau Toba (Samosir, Parapat,
  Tomok, Tongging, Bukit Holbung, Silangit, Pusuk Buhit)

---

## 3. Aturan Labeling

| Label   | Kriteria Penentuan |
|---------|--------------------|
| positif | Mengandung apresiasi, pujian, ekspresi kepuasan, rekomendasi eksplisit, atau rasa kagum terhadap aspek wisata |
| netral  | Deskriptif atau informatif tanpa emosi kuat; penilaian seimbang; kepuasan terkualifikasi; kondisi standar |

**Aturan tiebreaker:**
- Jika kalimat mengandung aspek positif dan catatan minor → **positif**
- Jika penilaian semua aspek rata-rata (oke/lumayan/cukup) → **netral**
- Jika ada ekspresi keinginan kembali atau rekomendasi → **positif**

---

## 4. Distribusi Topik (kategori_keluhan)

| Kategori    | Jumlah | Target % |
|-------------|--------|----------|
| kebersihan  | ~63    | 18%      |
| pelayanan   | ~63    | 18%      |
| fasilitas   | ~56    | 16%      |
| akses       | ~42    | 12%      |
| keamanan    | ~35    | 10%      |
| pemandangan | ~35    | 10%      |
| harga       | ~28    | 8%       |
| kuliner/budaya/lainnya | ~28 | 8% |

---

## 5. Validasi Kualitas Data

- ✅ Tidak ada baris kosong pada kolom ulasan/label
- ✅ Label hanya: positif/netral (tidak ada label invalid)
- ✅ Distribusi label tepat: 250 positif, 100 netral
- ✅ Tidak ada karakter rusak/encoding error
- ✅ Semua file dapat dibuka normal di Excel
- ✅ Duplikasi: 0 baris (0%)
- ✅ Slang ≤ 10% dari total data

---

## 6. Struktur File Output

| File | Isi |
|------|-----|
| `1_dataset_sentimen_toba_positif.xlsx` | Dataset utama: 350 baris, warna per label |
| `2_lexicon_positif.xlsx`              | 5 sheet lexicon kata/frasa positif |
| `3_hard_cases_positif.xlsx`           | 30 kasus ambigu + analisis label |
| `4_ringkasan_statistik_positif.xlsx`  | Distribusi label, kategori, statistik teks |
| `5_README_positif.md`                 | Dokumentasi ini |

---

## 7. Panduan Merge ke Pipeline Existing

Dataset ini dirancang kompatibel dengan **paket negatif** yang telah dibuat sebelumnya.
Skema kolom identik: `No | ulasan | label | kategori_keluhan | sumber | catatan`.

```python
import pandas as pd

df_negatif = pd.read_excel("1_dataset_sentimen_toba.xlsx")
df_positif = pd.read_excel("1_dataset_sentimen_toba_positif.xlsx")

# Reset nomor urut sebelum concat
df_merged = pd.concat([df_negatif, df_positif], ignore_index=True)
df_merged["No"] = range(1, len(df_merged) + 1)

df_merged.to_excel("dataset_merged_final.xlsx", index=False)
```

Distribusi final setelah merge:
- Negatif: 300 baris
- Netral: 150 baris (50 dari paket 1 + 100 dari paket 2)
- Positif: 300 baris (50 dari paket 1 + 250 dari paket 2)
- **Total: 750 baris**

---

## 8. Known Limitations

1. **Seluruh data adalah sintetik** — distribusi tidak menjamin representasi alami pengguna nyata
2. **Bahasa terlalu "terjaga"** — ulasan nyata sering mengandung typo dan singkatan yang lebih kasar
3. **Bias positif dalam hard cases** — 30 hard cases lebih banyak ke arah ambiguitas positif-netral
4. **Tidak mencakup dialek regional** — tidak ada campuran bahasa Batak, Minang, atau dialek lain

---

## 9. Rekomendasi Penggunaan

1. Gunakan sebagai **data augmentasi** bersama data scraping nyata
2. Prioritaskan data nyata dari **Google Maps dan TripAdvisor** untuk training utama
3. Gunakan **hard cases** sebagai test set untuk evaluasi rutin
4. Validasi **lexicon positif** dengan native speaker lokal Batak

---

## 10. Versi & Metadata

- Versi: 1.0
- Tanggal: April 2026
- Proyek: Smart Tourism Wisata Toba — Proyek Akhir TRPL IT Del
- Paket: 2/2 (paket ini = positif+netral; paket 1 = negatif)
