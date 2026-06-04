import os
import sys
import pandas as pd
from pathlib import Path

# Ensure the root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentiment_model.sentiment_model import SentimentAnalysisService

def run_test_cases():
    # Daftar test cases yang menantang
    test_cases = [
        "Aplikasi ini sangat bagus dan mudah digunakan, rekomended banget!",
        "Tolong dong aplikasinya diperbaiki, error terus pas mau login, sangat mengecewakan",
        "Biasa saja sih, tidak ada yang istimewa tapi lumayan buat dipakai sehari hari",
        "Agak kotor tempatnya, pelayanannya lama, dan harganya kemahalan",
        "Kamarnya bersih banget, tapi makanannya sangat tidak enak dan pelayanannya buruk", # Mixed sentimen konflik (seharusnya netral)
        "Mntp bgt apk ny", # Typos pendek
        "Sangat disarankan untuk keluarga",
        "Jelek bgt jgn didownload nyesel",
        "Awalnya bagus tapi lama kelamaan jadi lemot parah",
        "Wah gila sih ini, tempatnya kece abis dan worth it parah", # Kata gaul
        "Tolong perbaiki bug nya",
        "Pemandangannya indah",
    ]

    print("Loading model V11...")
    model_dir = PROJECT_ROOT / "sentiment_model" / "model_artifacts_v11"
    service = SentimentAnalysisService(model_dir=str(model_dir))
    
    print("Running predictions...")
    results = []
    for text in test_cases:
        res = service.predict_single(text=text)
        data = res['data']
        results.append({
            "Original Text": text,
            "Cleaned Text": data['processed_text'],
            "Final Prediction": data['label'],
            "Hybrid Reason": data['reason'],
            "Pos Prob": round(data['scores']['positive'], 3),
            "Neu Prob": round(data['scores']['neutral'], 3),
            "Neg Prob": round(data['scores']['negative'], 3),
        })

    df = pd.DataFrame(results)
    
    output_path = PROJECT_ROOT / "test_cases_v11_results.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Test cases saved to {output_path}")
    
    # Print summary to console
    print("\nSummary Results:")
    print(df[['Original Text', 'Final Prediction', 'Hybrid Reason']])

if __name__ == "__main__":
    run_test_cases()
