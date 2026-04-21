"""
Gabungkan semua file .xlsx di folder Merged_Excel menjadi satu file.

Cara pakai:
    python merge_xlsx_merged_excel.py

Opsional (atur folder dan nama output):
    python merge_xlsx_merged_excel.py --input-folder Merged_Excel --output-file hasil_gabungan.xlsx
"""

from pathlib import Path
import argparse

import pandas as pd


def gabungkan_semua_xlsx(
    input_folder: str = "Merged_Excel",
    output_file: str = "merged_all.xlsx",
) -> Path:
    folder = Path(input_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder.resolve()}")

    file_output = folder / output_file if not Path(output_file).is_absolute() else Path(output_file)

    daftar_file = []
    for file_xlsx in sorted(folder.glob("*.xlsx")):
        nama_file = file_xlsx.name.lower()
        if nama_file.startswith("~$"):
            continue
        if file_xlsx.resolve() == file_output.resolve():
            continue
        if nama_file.startswith("merged_all"):
            continue
        daftar_file.append(file_xlsx)

    if not daftar_file:
        raise FileNotFoundError(f"Tidak ada file .xlsx di folder: {folder.resolve()}")

    semua_df = []

    for file_xlsx in daftar_file:
        # Baca semua sheet agar tidak ada data yang terlewat
        semua_sheet = pd.read_excel(file_xlsx, sheet_name=None)

        for nama_sheet, df_sheet in semua_sheet.items():
            if df_sheet.empty:
                continue

            df_sheet = df_sheet.copy()
            df_sheet["sumber_file"] = file_xlsx.name
            df_sheet["sumber_sheet"] = nama_sheet
            semua_df.append(df_sheet)

    if not semua_df:
        raise ValueError("Semua sheet kosong, tidak ada data untuk digabungkan.")

    hasil = pd.concat(semua_df, ignore_index=True, sort=False)

    file_output.parent.mkdir(parents=True, exist_ok=True)
    hasil.to_excel(file_output, index=False)

    print("=" * 60)
    print("Penggabungan selesai")
    print(f"Jumlah file dibaca : {len(daftar_file)}")
    print(f"Jumlah baris hasil : {len(hasil)}")
    print(f"File output        : {file_output.resolve()}")
    print("=" * 60)

    return file_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gabungkan semua file XLSX di dalam folder Merged_Excel"
    )
    parser.add_argument(
        "--input-folder",
        default="Merged_Excel",
        help="Folder tempat file .xlsx berada (default: Merged_Excel)",
    )
    parser.add_argument(
        "--output-file",
        default="merged_all.xlsx",
        help="Nama file output (default: merged_all.xlsx)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gabungkan_semua_xlsx(args.input_folder, args.output_file)
