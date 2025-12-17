import os
import json
import random
import pandas as pd
from train_model import extract_features
from pathlib import Path

TRACK_NAMES = [
    "Summer Vibes", "Midnight Blues", "Ocean Waves", "Golden Hour",
    "Urban Nights", "Silent Echo", "Sunrise Beat", "Neon Dreams"
]

ARTISTS = [
    "DJ Nova", "Soul Singer", "Happy Tunes", "Waveform",
    "Echo Beats", "Lofi Collective", "Night Runner"
]

def create_audio_catalog(
    audio_dir: str,
    labels_csv: str,
    output_json: str = "catalog/catalog.json"
):
    ROOT_DIR = Path(__file__).resolve().parent
    df = pd.read_csv(ROOT_DIR / labels_csv)

    catalog = []
    track_id = 1

    for _, row in df.iterrows():
        filename = row["filename"]
        mood = row["mood"]

        audio_path = os.path.join(ROOT_DIR / audio_dir, filename)

        if not os.path.exists(audio_path):
            print(f"⚠️ File not found: {filename}")
            continue

        features, bpm = extract_features(audio_path)

        if features is None or bpm is None:
            continue

        catalog.append({
            "id": track_id,
            "name": random.choice(TRACK_NAMES),
            "artist": random.choice(ARTISTS),
            "mood": mood,
            "bpm": int(bpm),
            "features": features.tolist()
        })

        track_id += 1

    with open(ROOT_DIR / output_json, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"✅ Catalog created with {len(catalog)} tracks → {ROOT_DIR / output_json}")

if __name__ == "__main__":
    create_audio_catalog(
        audio_dir="datasets/audio_catalog",
        labels_csv="datasets/labels.csv",
        output_json="catalog/catalog.json"
    )
