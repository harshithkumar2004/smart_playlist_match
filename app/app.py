import numpy as np
import gradio as gr
import sqlite3
from datetime import datetime
import json
from train_model import extract_features, MoodClassifier
from pathlib import Path
import matplotlib.pyplot as plt
import librosa.display
import io

ROOT_DIR = Path(__file__).resolve().parent

db_dir = ROOT_DIR / "db"
db_dir.mkdir(parents=True, exist_ok=True)
DB_PATH = ROOT_DIR / 'db/playlist_matcher.db'

# ============= DATABASE SETUP =============
def init_db():
    """Initialize SQLite database for query logging"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS queries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  detected_mood TEXT,
                  detected_bpm REAL,
                  confidence REAL,
                  recommended_tracks TEXT)''')
    conn.commit()
    conn.close()

def log_query(mood, bpm, confidence, recommendations):
    """Log query to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO queries VALUES (NULL, ?, ?, ?, ?, ?)",
              (datetime.now().isoformat(), mood, bpm, confidence, 
               json.dumps([r['name'] for r in recommendations])))
    conn.commit()
    conn.close()




# def create_sample_catalog():
#     """Create a sample catalog with synthetic data"""
#     catalog = [
#         {'id': 1, 'name': 'Summer Vibes', 'artist': 'DJ Sunny', 'mood': 'happy', 'bpm': 120, 'features': None},
#         {'id': 2, 'name': 'Midnight Blues', 'artist': 'Soul Singer', 'mood': 'sad', 'bpm': 80, 'features': None},
#         {'id': 3, 'name': 'Workout Pump', 'artist': 'Fitness Beats', 'mood': 'energetic', 'bpm': 140, 'features': None},
#         {'id': 4, 'name': 'Zen Garden', 'artist': 'Meditation Master', 'mood': 'calm', 'bpm': 70, 'features': None},
#         {'id': 5, 'name': 'Happy Dance', 'artist': 'Pop Artist', 'mood': 'happy', 'bpm': 128, 'features': None},
#         {'id': 6, 'name': 'Rainy Day', 'artist': 'Acoustic Duo', 'mood': 'sad', 'bpm': 75, 'features': None},
#         {'id': 7, 'name': 'Power Hour', 'artist': 'Rock Band', 'mood': 'energetic', 'bpm': 145, 'features': None},
#         {'id': 8, 'name': 'Ocean Waves', 'artist': 'Nature Sounds', 'mood': 'calm', 'bpm': 65, 'features': None},
#         {'id': 9, 'name': 'Sunshine Pop', 'artist': 'Happy Tunes', 'mood': 'happy', 'bpm': 115, 'features': None},
#         {'id': 10, 'name': 'Melancholy', 'artist': 'Sad Songs Inc', 'mood': 'sad', 'bpm': 72, 'features': None},
#     ]

#     # Generate synthetic features for catalog tracks
#     for track in catalog:
#         # Create mood-based synthetic features
#         if track['mood'] == 'happy':
#             base = np.random.randn(30) * 0.3 + [0.5] * 30
#         elif track['mood'] == 'sad':
#             base = np.random.randn(30) * 0.3 - [0.3] * 30
#         elif track['mood'] == 'energetic':
#             base = np.random.randn(30) * 0.5 + [0.8] * 30
#         else:  # calm
#             base = np.random.randn(30) * 0.2
#         track['features'] = base

#     return catalog

def create_sample_catalog(catalog_path="catalog/catalog.json"):
    """
    Load catalog from JSON and return it
    """

    with open(ROOT_DIR / catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    for track in catalog:
        # Convert features list → numpy array
        if track.get("features") is not None:
            track["features"] = np.array(track["features"], dtype=float)

        # Safety: enforce BPM type
        track["bpm"] = int(track["bpm"])

        # Safety: normalize mood
        track["mood"] = track["mood"]

    return catalog

# ============= MATCHING LOGIC =============
def find_matches(query_features, query_mood, query_bpm, catalog, 
                 mood_weight=0.7, tempo_weight=0.3, top_k=5):
    """Find matching tracks from catalog"""
    if isinstance(query_bpm, np.ndarray):
        query_bpm = float(query_bpm.item())
    
    matches = []
    
    for track in catalog:
        # Mood match (binary: 1 if match, 0 otherwise)
        mood_match = 1.0 if track['mood'] == query_mood else 0.0
        
        # Tempo similarity (within ±8% BPM is considered similar)
        tempo_diff = abs(track['bpm'] - query_bpm)
        if isinstance(tempo_diff, np.ndarray):
            tempo_diff = float(tempo_diff.item())
            
        tempo_threshold = query_bpm * 0.08
        if tempo_diff <= tempo_threshold:
            tempo_similarity = 1.0 - (tempo_diff / tempo_threshold)
        else:
            tempo_similarity = max(0, 1.0 - (tempo_diff / 50))
        
        # Feature distance (euclidean distance)
        if query_features is not None and track['features'] is not None:
            feature_dist = np.linalg.norm(query_features - track['features'])
            if isinstance(feature_dist, np.ndarray):
                feature_dist = float(feature_dist.item())
            feature_similarity = 1.0 / (1.0 + feature_dist)
        else:
            feature_similarity = 0.5
        
        # Combined score
        score = (mood_match * mood_weight + 
                tempo_similarity * tempo_weight + 
                feature_similarity * (1 - mood_weight - tempo_weight))
        
        matches.append({
            'id': track['id'],
            'name': track['name'],
            'artist': track['artist'],
            'mood': track['mood'],
            'bpm': int(track['bpm']),
            'score': float(score),
            'tempo_diff': float(tempo_diff),
            'mood_match':  bool(mood_match)
        })
    
    # Sort by score and return top K
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches[:top_k]

# ============= GRADIO UI =============
def analyze_and_recommend(audio_file, mood_weight, tempo_weight):
    """Main function to analyze audio and recommend tracks"""
    print("Analyzing...")
    
    if audio_file is None:
        return "Please upload an audio file", ""
    # Extract features
    features, bpm = extract_features(audio_file)
    print("input audio features len", len(features), bpm)
    
    if features is None:
       return "Error processing audio file", ""
    
    # Predict mood
    classifier = MoodClassifier()
    classifier.load()  # Try to load pre-trained model
    
    # If not trained, use rule-based prediction
    if not classifier.is_trained:
        # Simple rule-based mood prediction from energy and tempo
        energy = features[-1]  # RMS energy
        if bpm > 130 and energy > 0.1:
            mood = 'energetic'
            confidence = 0.75
        elif bpm < 80 and energy < 0.08:
            mood = 'sad'
            confidence = 0.70
        elif bpm > 110 and energy > 0.08:
            mood = 'happy'
            confidence = 0.72
        else:
            mood = 'calm'
            confidence = 0.68
        print("used rule-based model: ", mood, confidence)
    else:
        mood, confidence = classifier.predict(features)
        print("used pre-trained model: ", mood, confidence)
    # ---------- NOT CONFIDENT FALLBACK ----------
    CONF_THRESHOLD = 0.7
    if confidence < CONF_THRESHOLD:
        mood = "not confident"
        mood_weight = 0.0   # ignore mood in ranking

    # Load catalog
    catalog = create_sample_catalog()
    # print("Catalog", catalog)
    
    # Find matches
    recommendations = find_matches(features, mood, bpm, catalog, 
                                   mood_weight, tempo_weight)
    print("recommendations", recommendations)
    
    # Log query
    log_query(mood, bpm, confidence, recommendations)
    
    # Format results
    mood_emoji = {
    'happy': '😊',
    'sad': '😢',
    'energetic': '⚡',
    'calm': '🧘',
    'not confident': '❓'
}

    
    analysis_text = f"""
        ### 🎵 Audio Analysis Results

        **Detected Mood:** {mood_emoji.get(mood, '🎵')} {mood.upper()}
        **Tempo (BPM):** {bpm}
        **Confidence:** {confidence*100:.1f}%
    """
    
    if confidence < 0.7:
        analysis_text += "\n\n⚠️ **Low Confidence Warning:** Detection confidence is below 70%. Results may be less accurate." 

  
    
    # Format recommendations
    rec_text = "### 🎧 Top 5 Recommendations\n\n"
    for i, rec in enumerate(recommendations, 1):
        rec_text += f"""
                **{i}. {rec['name']}** by {rec['artist']}
                - Mood: {mood_emoji.get(rec['mood'], '🎵')} {rec['mood'].capitalize()}
                - BPM: {rec['bpm']} (Δ {rec['tempo_diff']:.0f} BPM)
                - Match Score: {rec['score']*100:.2f}%
                - Mood Match: {'✓' if rec['mood_match'] else '✗'}

                ---
            """
    
    print(f"Query logged at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return analysis_text, rec_text 

# ============= MAIN APP =============
def create_app():
    """Create Gradio interface"""
    
    # Initialize database
    init_db()
    
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎵 Smart Playlist Matcher
        Upload an audio clip and get personalized track recommendations based on mood and tempo!
        """)
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Upload Audio File (10-60s clip)",
                    type="filepath"
                )
                
                with gr.Row():
                    mood_weight = gr.Slider(
                        0, 1, value=0.7, step=0.1,
                        label="Mood Weight"
                    )
                    tempo_weight = gr.Slider(
                        0, 1, value=0.3, step=0.1,
                        label="Tempo Weight"
                    )
                
                analyze_btn = gr.Button("🎯 Analyze & Find Matches", variant="primary")
        
        with gr.Row():
            with gr.Column():
                analysis_output = gr.Markdown(label="Analysis Results")
            with gr.Column():
                recommendations_output = gr.Markdown(label="Recommendations")
        
        analyze_btn.click(
            fn=analyze_and_recommend,
            inputs=[audio_input, mood_weight, tempo_weight],
            outputs=[analysis_output, recommendations_output]
        )
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(share=True)
