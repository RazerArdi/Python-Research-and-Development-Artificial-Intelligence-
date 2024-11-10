# 1. Import libraries yang diperlukan
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 2. Membaca dataset
data_path = r"Dataset/tcc_ceds_music.csv"
music_data = pd.read_csv(data_path)

# 3. Menampilkan informasi awal dataset
print(music_data.head())
print(music_data.info())

# 4. Memilih kolom-kolom fitur numerik yang akan digunakan untuk rekomendasi
feature_columns = ['danceability', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'energy']
music_features = music_data[feature_columns]

# 5. Menyiapkan model KNN untuk mencari lagu serupa
kNN_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
kNN_model.fit(music_features)

# 6. Membuat dictionary untuk mapping indeks ke nama lagu dan artis
track_mapper = dict(enumerate(music_data['artist_name'] + " - " + music_data['track_name']))

# 7. Fungsi untuk mencari rekomendasi lagu serupa
def find_similar_tracks(track_name, artist_name, k=5):
    track_identifier = f"{artist_name} - {track_name}"
    
    # Mencari indeks dari lagu yang diinginkan
    try:
        track_index = list(track_mapper.values()).index(track_identifier)
    except ValueError:
        print(f"Track '{track_name}' by '{artist_name}' not found in dataset.")
        return
    
    # Mendapatkan vektor fitur dari lagu
    track_vec = music_features.iloc[track_index].values.reshape(1, -1)
    distances, indices = kNN_model.kneighbors(track_vec, n_neighbors=k+1)
    
    # Menampilkan hasil rekomendasi (mengecualikan lagu itu sendiri)
    print(f"Since you listened to '{track_identifier}', you might also like:")
    for i in range(1, k+1):
        similar_track_index = indices.flatten()[i]
        print(f"- {track_mapper[similar_track_index]}")

# 8. Contoh penggunaan: mencari rekomendasi untuk lagu tertentu
find_similar_tracks("i believe", "frankie laine", k=5)
