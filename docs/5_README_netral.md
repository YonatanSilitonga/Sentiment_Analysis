# README — Dataset Sentimen Netral Wisata Danau Toba

## Deskripsi Paket
Paket dataset ini berisi **200 ulasan berbahasa Indonesia** dengan label **netral** untuk domain wisata Danau Toba, mencakup kawasan Balige, Samosir, Parapat, Pangururan, Tuktuk, dan sekitarnya. Dataset ini dirancang untuk memperkuat performa kelas NETRAL pada model klasifikasi sentimen 3 kelas (negatif, netral, positif).

---

## Daftar File

| File | Deskripsi | Jumlah Baris |
|------|-----------|--------------|
| `1_dataset_sentimen_toba_netral.xlsx` | Dataset utama 200 ulasan netral | 200 |
| `2_lexicon_netral.xlsx` | Kamus kata/frasa netral (6 sheet tematik) | ~128 entri |
| `3_hard_cases_netral.xlsx` | Kalimat ambigu yang rawan salah klasifikasi | 40 |
| `4_ringkasan_statistik_netral.xlsx` | Statistik deskriptif dan distribusi | — |
| `5_README_netral.md` | Dokumentasi ini | — |

---

## Skema Dataset Utama (File 1)

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `No` | Integer | Nomor urut baris, mulai dari 1 |
| `ulasan` | String | Teks ulasan berbahasa Indonesia |
| `label` | String | Selalu `netral` |
| `kategori_keluhan` | String | Topik utama ulasan |
| `sumber` | String | Platform asal (Google Maps / TripAdvisor) |
| `catatan` | String | Jenis konstruksi kalimat netral |

---

## Metode Pembuatan Data

Dataset ini dibuat secara **sintetis kuasi-natural** dengan pendekatan berikut:

1. **Identifikasi topik**: Berdasarkan 7 kategori keluhan yang umum di ulasan wisata Toba.
2. **Template semantik**: Setiap kalimat disusun menggunakan pola linguistik netral (deskriptif, kontras ringan, atau evaluasi moderat).
3. **Variasi linguistik**: Panjang kalimat bervariasi (pendek 8–12 kata, sedang 15–20 kata, panjang 25+ kata).
4. **Distribusi sumber**: ~60% disimulasikan sebagai Google Maps, ~40% TripAdvisor.

---

## Kriteria Labeling Netral

Sebuah ulasan dikategorikan **netral** jika memenuhi salah satu kondisi berikut:

- **Deskriptif murni**: Menyampaikan fakta atau kondisi tanpa evaluasi emosional (contoh: _"Feri beroperasi setiap 2 jam"_).
- **Evaluasi moderat**: Menggunakan kata kunci moderat seperti _cukup, lumayan, standar, relatif, agak, tidak terlalu_.
- **Kontras ringan**: Memiliki sisi positif dan negatif yang seimbang dengan kata penghubung _tapi, namun, hanya saja, walaupun, meski_.
- **Kondisional**: Pernyataan yang hanya berlaku dalam kondisi tertentu (cuaca, musim, waktu).

### Kalimat yang BUKAN netral (dihindari):
- Positif kuat: _"Sangat puas", "luar biasa", "terbaik", "wajib dikunjungi"_
- Negatif kuat: _"Sangat mengecewakan", "parah", "tidak akan kembali", "rugi datang ke sini"_

---

## Distribusi Kategori Target

| Kategori | Target | Realisasi |
|----------|--------|-----------|
| akses | 20% (40 baris) | 40 baris |
| fasilitas | 20% (40 baris) | 40 baris |
| kebersihan | 15% (30 baris) | 30 baris |
| pelayanan | 15% (30 baris) | 30 baris |
| keamanan | 10% (20 baris) | 20 baris |
| harga | 10% (20 baris) | 20 baris |
| pemandangan/lainnya | 10% (20 baris) | 20 baris |

---

## Validasi Kualitas

- ✅ Tidak ada nilai null di kolom `ulasan` dan `label`
- ✅ Semua 200 baris berlabel `netral`
- ✅ Tidak ada duplikasi literal antar baris
- ✅ Format Excel valid dan kompatibel dengan openpyxl/pandas
- ✅ Bahasa natural Indonesia, tidak formulaik atau template robotik
- ✅ Estimasi penggunaan slang < 10%
- ✅ Nama kolom identik dengan dataset sebelumnya

---

## Keterbatasan Data

1. **Data sintetis**: Meskipun disusun natural, data ini tidak berasal dari scraping nyata. Distribusi linguistiknya mungkin tidak sepenuhnya merepresentasikan variasi bahasa pengguna asli.
2. **Bias domain**: Semua kalimat terfokus pada kawasan Danau Toba — tidak cocok digunakan sebagai data generalis di luar domain wisata Toba.
3. **Tidak ada verifikasi native speaker**: Beberapa ekspresi mungkin terdengar sedikit formal dibanding ulasan spontan pengguna.
4. **Slang terbatas**: Sengaja dijaga di bawah 10%, sehingga mungkin kurang merepresentasikan gaya tulisan informal generasi muda.

---

## Kompatibilitas Pipeline

Dataset ini siap digabungkan dengan dataset sentimen Toba yang sudah ada:

```python
import pandas as pd

existing = pd.read_excel("dataset_existing.xlsx")
new_neutral = pd.read_excel("1_dataset_sentimen_toba_netral.xlsx")

# Gabungkan
combined = pd.concat([existing, new_neutral], ignore_index=True)
combined["No"] = range(1, len(combined) + 1)
combined.to_excel("dataset_combined.xlsx", index=False)
```

Kolom yang tersedia: `No`, `ulasan`, `label`, `kategori_keluhan`, `sumber`, `catatan`

---

## Kontak & Versi

- **Versi**: 1.0
- **Tanggal**: April 2026
- **Domain**: Wisata Danau Toba, Sumatera Utara
- **Bahasa**: Indonesia
- **Label kelas**: netral (3-class: negatif / netral / positif)
