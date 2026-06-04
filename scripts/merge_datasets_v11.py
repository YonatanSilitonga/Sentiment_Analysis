import os
import sys
import pandas as pd
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DIR = PROJECT_ROOT / "Merged_Excel"
MERGED_DIR.mkdir(exist_ok=True)

v3_path = MERGED_DIR / "dataset_labeled_combined_v3.xlsx"
tsv_path = PROJECT_ROOT / "train_preprocess_ori.tsv"
hotel_path = PROJECT_ROOT / "hotel_sentimen_berlabel.xlsx"
output_path = MERGED_DIR / "dataset_v11_master.xlsx"

def merge_datasets():
    print("Loading base dataset (V3)...")
    if v3_path.exists():
        df_base = pd.read_excel(v3_path)
    else:
        print(f"Warning: Base dataset {v3_path} not found. Creating empty dataframe.")
        df_base = pd.DataFrame(columns=['nama', 'tanggal', 'ulasan', 'label', 'No', 'kategori_keluhan', 'sumber', 'catatan'])

    print(f"Base dataset shape: {df_base.shape}")

    dfs_to_concat = [df_base]

    # Load TSV
    print("Loading TSV dataset...")
    if tsv_path.exists():
        df_tsv = pd.read_csv(tsv_path, sep='\t')
        # Mapping sentiment: positive -> positif, negative -> negatif
        label_map = {'positive': 'positif', 'neutral': 'netral', 'negative': 'negatif'}
        df_tsv['label'] = df_tsv['sentiment'].str.lower().map(label_map).fillna(df_tsv['sentiment'].str.lower())
        df_tsv = df_tsv.rename(columns={'text': 'ulasan'})
        df_tsv['sumber'] = 'train_preprocess_ori.tsv'
        
        # Keep only required columns
        df_tsv = df_tsv[['ulasan', 'label', 'sumber']]
        dfs_to_concat.append(df_tsv)
        print(f"TSV dataset shape: {df_tsv.shape}")
    else:
        print(f"Warning: TSV dataset {tsv_path} not found.")

    # Load Hotel Excel
    print("Loading Hotel dataset...")
    if hotel_path.exists():
        df_hotel = pd.read_excel(hotel_path)
        df_hotel = df_hotel.rename(columns={'Ulasan': 'ulasan', 'Sentimen': 'label'})
        if 'label' in df_hotel.columns:
            df_hotel['label'] = df_hotel['label'].astype(str).str.lower()
        df_hotel['sumber'] = 'hotel_sentimen_berlabel.xlsx'
        
        cols = [c for c in ['ulasan', 'label', 'sumber'] if c in df_hotel.columns]
        df_hotel = df_hotel[cols]
        dfs_to_concat.append(df_hotel)
        print(f"Hotel dataset shape: {df_hotel.shape}")
    else:
        print(f"Warning: Hotel dataset {hotel_path} not found.")

    print("Concatenating datasets...")
    df_master = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Filter valid labels just in case
    valid_labels = ['positif', 'netral', 'negatif']
    df_master = df_master[df_master['label'].isin(valid_labels)]
    
    print(f"Master dataset final shape: {df_master.shape}")
    print("Saving to Excel...")
    df_master.to_excel(output_path, index=False)
    print(f"Successfully saved master dataset to {output_path}")

if __name__ == "__main__":
    merge_datasets()
