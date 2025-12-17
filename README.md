# Smart Playlist Matcher

A local audio analysis and recommendation system that matches uploaded audio clips with similar tracks from a catalog based on mood and tempo.

##  Features

- **Audio Analysis**: Extracts MFCCs, tempo (BPM), spectral features, and energy using librosa
- **Mood Classification**: Predicts mood (happy, sad, serious, calm) using a trained MLP classifier
- **Smart Matching**: Recommends top 5 similar tracks based on mood, tempo, and audio features
- **Interactive UI**: Clean Gradio interface with adjustable mood/tempo weights
- **Query Logging**: Stores all queries in local SQLite database
- **Visualization**: Shows confidence scores
- **Local Processing**: Everything runs on your machine, no external APIs needed

##  Bonus Features

- **Mood–Tempo Weight Slider**: Lets users control how shown mood vs. tempo affects ranking.
- **Low-Confidence Fallback**: Displays *“Not Confident”* when mood prediction confidence is below a threshold.

## Quick Start

### Installation

```bash
# Clone the repository
git clone <git-repo-url>
cd smart-playlist-matcher

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
# Option 1: Train model first (recommended)
python .\app\train_model.py --audio-dir datasets/audio_catalog --labels datasets/labels.csv

# Option 2: Run app (will use rule-based classification if no model exists)
python app.py
```

The app will launch at `http://localhost:7860`

## Requirements

Create a `requirements.txt` file with:

```
numpy
librosa
scikit-learn
gradio
soundfile
```

## Usage

### 1. Upload Audio
- Click "Upload Audio File" and select a 10-60 second audio clip
- Supported formats: MP3, WAV, FLAC, OGG, etc.

### 2. Adjust Weights (Optional)
- **Mood Weight**: How much to prioritize mood matching (default: 70%)
- **Tempo Weight**: How much to prioritize tempo similarity (default: 30%)
- Weights automatically adjust to sum to 100%

### 3. Analyze
- Click "Analyze & Find Matches"
- View detected mood, BPM, and confidence
- Get 5 recommended tracks with similarity scores

### 4. Review Results
- **Analysis Results**: Shows detected mood, tempo, and confidence
- **Recommendations**: Top 5 matching tracks with details
- **Features**: Extracted MFCC coefficients
- **Log Status**: Confirmation of database logging



##  Project Structure

```

smart-playlist-app/
├── app.py                      # Main Gradio application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── models/
│   └── mood_classifier.pkl     # Trained model (generated)
├── db/
│   └── playlist_matcher.db     # SQLite database (generated)
├── datasets/
│   ├── audio_catalog/          # Optional: catalog audio
│   │   ├── track1.mp3
│   │   ├── track2.mp3
│   │   └── ...
│   └── labels.csv

```

##  Training with Your Own Data

### Step 1: Prepare Audio Files

Create a directory with your audio files:

```
datasets/audio_catalog/
├── happy_song1.mp3
├── sad_song1.mp3
├── energetic_song1.mp3
├── calm_song1.mp3
└── ...
```

### Step 2: Create Labels File

Create `labels.csv` with format:

```csv
filename,mood
happy_song1.mp3,happy
sad_song1.mp3,sad
energetic_song1.mp3,energetic
calm_song1.mp3,calm
```

### Step 3: Train Model

```bash
python .\app\train_model.py --audio-dir datasets/audio_catalog --labels datasets/labels.csv
```

The model will be saved to `models/mood_classifier.pkl`.

## Adding Tracks to Catalog

To add tracks to the recommendation catalog:

```bash
python .\app\create_catalog.py
```
## Running the Project Locally

Firstly, download entire project files from the repo

```bash
cd smart-playlist-matcher
```

(Optional but recommended)

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model (first time only):

```bash
python app/train_model.py --audio-dir datasets/audio_catalog --labels datasets/labels.csv
```

Update catalog:

```bash
python .\app\create_catalog.py
```

Run the app:

```bash
python app/app.py
```
Open in browser:

```
http://localhost:7860
```
**NOTE:** While running the application, as soon as you open the Gradio interface, upload an audio file and click **Analyze**, check the terminal. It will display **“Analyzing…”**. This might take some time. If it takes too long, simply refresh the browser and upload an audio file and click **Analyze**. It will work . ( Refreshing is needed only for the first time ) 




##  How It Works

### Feature Extraction
The system extracts these features using librosa:
- **MFCCs**: 13 coefficients extracted and visualized 
- **Tempo**: BPM detection using beat tracking
- **Spectral Centroid**: Brightness of sound
- **Spectral Rolloff**: Frequency shape
- **Zero Crossing Rate**: Noisiness/percussiveness
- **RMS Energy**: Overall loudness

### Mood Classification
A Multi-Layer Perceptron (MLP) neural network with:
- Input layer: 29 features
- Hidden layers: 64 → 32 neurons
- Output layer: 4 moods (happy, sad, serious, calm)
- Activation: ReLU
- Trained using sklearn's MLPClassifier

### Matching Algorithm
Tracks are ranked by combined score:

```
score = (mood_match × mood_weight) + 
        (tempo_similarity × tempo_weight) + 
        (feature_similarity × remaining_weight)
```

Where:
- **mood_match**: 1.0 if mood matches, 0.0 otherwise
- **tempo_similarity**: Based on ±8% BPM threshold
- **feature_similarity**: Inverse of Euclidean distance



##  Database Schema

Queries are logged to `db/playlist_matcher.db`:

```sql
CREATE TABLE queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    detected_mood TEXT,
    detected_bpm REAL,
    confidence REAL,
    recommended_tracks TEXT  -- JSON array of track names
);
```

View query history:

```bash
sqlite3 playlist_matcher.db "SELECT * FROM queries ORDER BY timestamp DESC LIMIT 10;"
```

##  Acknowledgments

- Built with [Gradio](https://gradio.app/)
- Audio processing by [librosa](https://librosa.org/)
- Machine learning with [scikit-learn](https://scikit-learn.org/)

  
## Limitations:
- This system works best with short audio clips and basic audio characteristics, so it may not always capture deeper or more complex musical emotions.
- The accuracy of mood prediction depends heavily on how diverse and well-labeled the training data is.
- Tempo (BPM) detection can sometimes be inaccurate for songs that are ambient, free-flowing, or lack a clear rhythm.
- Additionally, the system is intended for small, locally stored music collections and is not currently optimized to handle very large music libraries.

## Future Enhancements:
- In the future, the system can be improved by using more advanced audio features to better understand music emotions. Adding more and varied training data will help increase the accuracy of mood prediction.
- Tempo detection can be improved to work better for songs that do not have a clear rhythm. The system can also be expanded to handle larger music collections more efficiently.
- Additional features such as user-based recommendations, better visual displays, and improved playback controls can be added to make the system more interactive and user-friendly.
- Additional visualizations such as spectrograms or beat-aligned tempo plots can be added.
