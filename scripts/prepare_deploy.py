import os
import shutil
from pathlib import Path

src_dir = Path(r"d:\semester-4-IT Del\Semester VI\UI-UX DESIGN\Sentiment_Analysis")
dest_dir = Path(r"d:\semester-4-IT Del\Semester VI\UI-UX DESIGN\sentiment_analysis_deploy")

print("Memulai pemindahan file ke folder deploy...")

if dest_dir.exists():
    print(f"Menghapus folder lama: {dest_dir}")
    shutil.rmtree(dest_dir)

dest_dir.mkdir(parents=True, exist_ok=True)

# File di root yang diperlukan
files_to_copy = [
    "app.py",
    "requirements.txt",
    "combined_slang_words.txt"
]

for file in files_to_copy:
    src_file = src_dir / file
    if src_file.exists():
        print(f"Menyalin file: {file}")
        shutil.copy2(src_file, dest_dir / file)

# Menyalin folder sentiment_model secara selektif
src_model_dir = src_dir / "sentiment_model"
dest_model_dir = dest_dir / "sentiment_model"
dest_model_dir.mkdir(exist_ok=True)

# Salin semua file Python di dalam folder sentiment_model
for item in src_model_dir.iterdir():
    if item.is_file() and item.suffix == ".py":
        print(f"Menyalin modul: sentiment_model/{item.name}")
        shutil.copy2(item, dest_model_dir / item.name)

# Salin hanya folder model v11
src_artifacts = src_model_dir / "model_artifacts_v11"
dest_artifacts = dest_model_dir / "model_artifacts_v11"
if src_artifacts.exists():
    print("Menyalin artefak model v11...")
    shutil.copytree(src_artifacts, dest_artifacts)

print("\n" + "=" * 50)
print(f"SELESAI! Folder deploy berhasil dibuat di:\n{dest_dir.resolve()}")
print("=" * 50)
