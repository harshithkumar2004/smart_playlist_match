import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from pathlib import Path

MOODS = ['happy', 'sad', 'serious', 'calm']

# ============= FEATURE EXTRACTION =============
def extract_features(audio_path, duration=30):
    """Extract audio features using librosa"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, duration=duration)
        
        # Extract MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Extract tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        # Energy/RMS
        rms = librosa.feature.rms(y=y)
        
        # Combine features
        features = np.concatenate([
            mfccs_mean,
            mfccs_std,
            [np.mean(spectral_centroids)],
            [np.mean(spectral_rolloff)],
            [np.mean(zero_crossing_rate)],
            [np.mean(rms)]
        ])
        
        return features, tempo
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None
    
# ============= MOOD CLASSIFIER =============
class MoodClassifier:
    """Simple MLP classifier for mood prediction"""
    
    def __init__(self):
        self.moods = MOODS
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            max_iter=500,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X, y):
        """Train the classifier"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, features):
        """Predict mood from features"""
        if not self.is_trained:
            # Return mock prediction if not trained
            return 'calm', 0.75
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        mood_idx = self.model.predict(features_scaled)[0]
        confidence = np.max(self.model.predict_proba(features_scaled))
        return self.moods[mood_idx], confidence
    
    def save(self, path='models/mood_classifier.pkl'):
        """Save model to disk"""
        ROOT_DIR = Path(__file__).resolve().parent
        models_dir = ROOT_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        with open(ROOT_DIR / path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 
                        'is_trained': self.is_trained}, f)
    
    def load(self, path='models/mood_classifier.pkl'):
        """Load model from disk"""
        ROOT_DIR = Path(__file__).resolve().parent
        if os.path.exists(ROOT_DIR / path):
            with open(ROOT_DIR / path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = data['is_trained']
                
def generate_synthetic_training_data(n_samples_per_mood=100):
    """
    Generate synthetic training data for mood classification
    In production, replace this with real labeled audio data
    """
    moods = MOODS
    X = []
    y = []

    for mood_idx, mood in enumerate(moods):
        for _ in range(n_samples_per_mood):
            # Generate synthetic 30-dimensional features
            # Feature patterns differ by mood
            if mood == 'happy':
                # Happy: higher spectral features, moderate energy
                features = np.random.randn(30) * 0.3 + np.array(
                    [0.5] * 13 + [0.3] * 13 + [0.6, 0.7, 0.05, 0.12]
                )
            elif mood == 'sad':
                # Sad: lower energy, lower spectral features
                features = np.random.randn(30) * 0.3 + np.array(
                    [-0.3] * 13 + [0.2] * 13 + [0.3, 0.4, 0.03, 0.06]
                )
            elif mood == 'serious':
                # Energetic: high energy, high spectral activity
                features = np.random.randn(30) * 0.5 + np.array(
                    [0.8] * 13 + [0.4] * 13 + [0.9, 0.95, 0.08, 0.18]
                )
            else:  # calm
                # Calm: low energy, stable features
                features = np.random.randn(30) * 0.2 + np.array(
                    [0.0] * 13 + [0.15] * 13 + [0.4, 0.5, 0.02, 0.08]
                )

            X.append(features)
            y.append(mood_idx)

    return np.array(X), np.array(y)

def train_from_audio_files(audio_dir, labels_file, aug_factor = 16):
    """
    Train from real audio files
    
    Args:
        audio_dir: Directory containing audio files
        labels_file: CSV file with format: filename,mood
    """
    import pandas as pd
    
    ROOT_DIR = Path(__file__).resolve().parent
    
    if not os.path.exists(ROOT_DIR / audio_dir):
        print(f"Error: Audio directory {ROOT_DIR / audio_dir} not found")
        return None, None
    
    if not os.path.exists(ROOT_DIR / labels_file):
        print(f"Error: Labels file {ROOT_DIR / labels_file} not found")
        return None, None
    
    # Load labels
    df = pd.read_csv(ROOT_DIR / labels_file)
    print(f"Original: {len(df)} samples")
    
    df = pd.concat([df] * aug_factor, ignore_index=True)
    print(f"Duplicated: {len(df)} samples")
    
    X = []
    y = []
    moods = MOODS
    
    print(f"Extracting features from {len(df)} audio files...")
    
    for idx, row in df.iterrows():
        audio_path = os.path.join(ROOT_DIR / audio_dir, row['filename'])
        mood = row['mood']
        
        if not os.path.exists(audio_path):
            print(f"Warning: {audio_path} not found, skipping...")
            continue
        
        try:
            features, _ = extract_features(audio_path)
            if features is not None:
                X.append(features)
                y.append(moods.index(mood))
                print(f"Processed {idx+1}/{len(df)}: {row['filename']}")
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    if len(X) == 0:
        print("Error: No features extracted")
        return None, None
    
    return np.array(X), np.array(y)

def train_and_evaluate(use_synthetic=True, audio_dir=None, labels_file=None):
    """Train and evaluate the mood classifier"""
    
    print("=" * 60)
    print("MOOD CLASSIFIER TRAINING")
    print("=" * 60)
    
    ROOT_DIR = Path(__file__).resolve().parent
    # Load or generate training data
    if use_synthetic:
        print("\n📊 Generating synthetic training data...")
        X, y = generate_synthetic_training_data(n_samples_per_mood=100)
        print(f"Generated {len(X)} synthetic samples")
    else:
        print(f"\n📁 Loading audio files from {ROOT_DIR / audio_dir}...")
        X, y = train_from_audio_files(audio_dir, labels_file, 3)
        if X is None:
            print("Failed to load training data. Falling back to synthetic data.")
            X, y = generate_synthetic_training_data(n_samples_per_mood=100)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📈 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Train classifier
    print("\n🧠 Training classifier...")
    classifier = MoodClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluate
    print("\n📋 Evaluating on test set...")
    X_test_scaled = classifier.scaler.transform(X_test)
    y_pred = classifier.model.predict(X_test_scaled)
    
    accuracy = np.mean(y_pred == y_test)
    print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=classifier.moods))
    
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("Predicted →")
    print("Actual ↓    ", "  ".join(f"{m[:4]:>6}" for m in classifier.moods))
    for i, mood in enumerate(classifier.moods):
        print(f"{mood[:8]:>8}    ", "  ".join(f"{cm[i,j]:>6}" for j in range(len(classifier.moods))))
    
    # Save model
    print("\n💾 Saving model...")
    classifier.save('models/mood_classifier.pkl')
    print("✅ Model saved to 'models/mood_classifier.pkl'")
    
    return classifier

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train mood classifier')
    parser.add_argument('--audio-dir', type=str, help='Directory with audio files')
    parser.add_argument('--labels', type=str, help='CSV file with labels (filename,mood)')
    parser.add_argument('--synthetic', action='store_true', 
                       help='Use synthetic data instead of real audio')
    
    args = parser.parse_args()
    
    if args.synthetic or (not args.audio_dir and not args.labels):
        print("Using synthetic training data")
        use_synthetic = True
        audio_dir = None
        labels_file = None
    else:
        use_synthetic = False
        audio_dir = args.audio_dir
        labels_file = args.labels
    
    # Train
    classifier = train_and_evaluate(use_synthetic, audio_dir, labels_file)
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()