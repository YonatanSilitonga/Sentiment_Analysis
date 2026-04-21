# README — Dataset Sentimen Wisata Toba

## 1. Gambaran Umum

Dataset ini dibuat untuk keperluan perbaikan model klasifikasi sentimen destinasi wisata
berbahasa Indonesia, dengan fokus domain Kawasan Wisata Danau Toba, Sumatera Utara.
Seluruh data bersifat sintetik/augmentasi — tidak berasal dari scraping platform nyata.

---

## 2. Cara Pengumpulan Data

- **Metode**: Generasi manual berbasis konteks domain
- **Pendekatan**: Setiap kalimat dibuat mengikuti pola bahasa alami pengguna Indonesia:
  formal, semi-formal, informal, slang, dan campur kode
- **Domain**: Ulasan destinasi wisata — khususnya konteks Danau Toba (Samosir, Parapat,
  Balige, dermaga, kawasan budaya Batak)
- **Variasi**: Setiap kategori mencakup variasi topik, gaya bahasa, dan tingkat eksplisitnya

---

## 3. Kriteria Pelabelan

| Label    | Kriteria                                                                 |
|----------|--------------------------------------------------------------------------|
| negatif  | Kalimat mengekspresikan keluhan, kekecewaan, atau pengalaman buruk       |
| netral   | Kalimat deskriptif, berimbang, atau tidak menyampaikan sentimen kuat     |
| positif  | Kalimat mengekspresikan kepuasan, pujian, atau pengalaman menyenangkan   |

**Catatan untuk kasus ambigu:**
- Jika kalimat mengandung elemen positif DAN negatif, label ditentukan oleh **sentimen dominan**
- Kalimat sarkasme menggunakan kata positif diklasifikasikan sebagai negatif
- Kalimat bersyarat ("kalau saja...") umumnya dikategorikan negatif karena mengandung kritik

---

## 4. Struktur File

| File | Isi |
|------|-----|
| `1_dataset_sentimen_toba.xlsx` | Dataset utama: ulasan, label, kategori, sumber |
| `2_lexicon_negatif.xlsx`       | Lexicon kata/frasa negatif per kategori (6 sheet) |
| `3_hard_cases.xlsx`            | 20 contoh kalimat ambigu + analisis kesulitan |
| `4_ringkasan_statistik.xlsx`   | Statistik distribusi label, kategori, deduplikasi |
| `5_README.md`                  | Dokumentasi ini |

---

## 5. Kategori Keluhan

| Kategori      | Deskripsi                                              |
|---------------|--------------------------------------------------------|
| kebersihan    | Kondisi toilet, sampah, bau, kebersihan fasilitas      |
| biaya_liar    | Pungli, calo, harga tidak wajar, kurang transparansi   |
| pelayanan     | Sikap petugas, respons pengelola, profesionalisme      |
| akses         | Jalan rusak, transportasi, sinyal, kemudahan mencapai  |
| fasilitas     | Kondisi fisik sarana prasarana, kelengkapan            |
| keamanan      | Penipuan, kehilangan, keselamatan, pengawasan          |
| lainnya       | Hal di luar kategori di atas                           |

---

## 6. Asumsi

1. Bahasa target utama: **Bahasa Indonesia** dengan toleransi terhadap slang, singkatan,
   dan campur kode (Jawa, Batak, Inggris)
2. Domain terbatas pada: **review destinasi wisata** (bukan review produk, film, dll)
3. Teks pendek hingga sedang (1–3 kalimat) sesuai pola review umum
4. Semua data berlabel `sumber: sintetik` — harus dikombinasikan dengan data nyata
   (hasil scraping) sebelum digunakan untuk pelatihan model production

---

## 7. Keterbatasan Data

- **Seluruh data adalah sintetik**: Tidak ada jaminan distribusi mencerminkan
  percakapan pengguna nyata
- **Bias pembuatan**: Kalimat mungkin terlalu "sempurna" secara tata bahasa
  dibanding ulasan nyata yang penuh typo dan singkatan tidak standar
- **Tidak mencakup semua dialek**: Ulasan dalam bahasa Batak, Minang, atau
  dialek regional lainnya tidak tersedia
- **Hard cases terbatas**: 20 contoh hard cases tidak mewakili semua pola ambiguitas
  yang mungkin muncul dalam data nyata

---

## 8. Rekomendasi Penggunaan

1. **Gunakan sebagai data augmentasi**, bukan data latih utama
2. **Gabungkan dengan data nyata** dari scraper Google Maps / TripAdvisor
3. **Validasi lexicon** dengan pakar domain atau native speaker
4. **Uji model pada hard cases** secara reguler untuk deteksi degradasi

---

## 9. Versi

- Versi: 1.0
- Tanggal pembuatan: April 2026
- Dibuat untuk: Smart Tourism Wisata Toba — Proyek Akhir TRPL IT Del
