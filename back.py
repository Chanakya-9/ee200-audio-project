import librosa
import numpy as np
import os
import pickle
from scipy.ndimage import maximum_filter
from collections import defaultdict

DB_PATH = "database.pkl"
SONG_DIR = "songs"

def create_hashes(peaks, fan_value=5):
    hashes = []
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if i + j < len(peaks):
                f1, t1 = peaks[i]
                f2, t2 = peaks[i + j]
                dt = t2 - t1
                if 0 < dt < 100:
                    hashes.append((f1, f2, dt, t1))
    return hashes

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    D = librosa.stft(y, n_fft=1024, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D))
    
    peaks = (maximum_filter(S_db, size=20) == S_db) & (S_db > -25)
    peak_coords = np.argwhere(peaks)
    peak_coords = peak_coords[peak_coords[:, 1].argsort()]
    
    return create_hashes(peak_coords), S_db, sr, peak_coords

def build_database():
    database = {}
    if not os.path.exists(SONG_DIR):
        os.makedirs(SONG_DIR)
        
    for file in os.listdir(SONG_DIR):
        if file.endswith(".mp3"):
            path = os.path.join(SONG_DIR, file)
            try:
                hashes, _, _, _ = extract_features(path)
                database[file] = hashes
            except:
                pass
                
    with open(DB_PATH, "wb") as f:
        pickle.dump(database, f)
    return database

def load_database():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "rb") as f:
                return pickle.load(f)
        except:
            os.remove(DB_PATH)
    return build_database()

database = load_database()

def build_index(database):
    index = defaultdict(list)
    for song, hashes in database.items():
        for h in hashes:
            index[(h[0], h[1], h[2])].append((song, h[3]))
    return index

index = build_index(database)