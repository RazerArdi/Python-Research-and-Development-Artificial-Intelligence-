import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class MusicRecommender:
    def __init__(self, music_data, feature_columns):
        self.music_data = music_data
        self.feature_columns = feature_columns
        self.model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
        self.track_mapper = dict(enumerate(music_data['artist_name'] + " - " + music_data['track_name']))
        self._fit_model()

    def _fit_model(self):
        """Training model KNN dengan data fitur musik."""
        music_features = self.music_data[self.feature_columns]
        self.model.fit(music_features)

    def find_similar_tracks(self, track_name, artist_name, k=5):
        """Mencari lagu serupa berdasarkan track name dan artist."""
        track_identifier = f"{artist_name} - {track_name}"
        try:
            track_index = list(self.track_mapper.values()).index(track_identifier)
        except ValueError:
            print(f"Track '{track_name}' by '{artist_name}' not found in dataset.")
            return
        
        track_vec = self.music_data[self.feature_columns].iloc[track_index].values.reshape(1, -1)
        distances, indices = self.model.kneighbors(track_vec, n_neighbors=k+1)

        recommendations = [self.track_mapper[indices.flatten()[i]] for i in range(1, k+1)]
        return recommendations
