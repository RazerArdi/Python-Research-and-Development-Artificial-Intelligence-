import data_loader
from recommender import MusicRecommender
import analyzer

data_path = r"Dataset/tcc_ceds_music.csv"

# Load dan cleaning data
music_data = data_loader.load_data(data_path)
music_data = data_loader.clean_data(music_data)

# Pilih kolom fitur
feature_columns = data_loader.get_feature_columns()

# Inisialisasi sistem rekomendasi
recommender = MusicRecommender(music_data, feature_columns)

# Mencari rekomendasi untuk lagu tertentu
track_name = "i believe"
artist_name = "frankie laine"
recommendations = recommender.find_similar_tracks(track_name, artist_name, k=5)

print(f"Since you listened to '{artist_name} - {track_name}', you might also like:")
for track in recommendations:
    print(f"- {track}")

# Analisis data
print("Missing values per column:\n", data_loader.get_missing_values(music_data))
print("Statistical description:\n", analyzer.describe_data(music_data, feature_columns))
print("Median values:\n", analyzer.calculate_median(music_data, feature_columns))

# Visualisasi
analyzer.plot_correlation_matrix(music_data, feature_columns)
analyzer.plot_danceability_by_genre(music_data)

# Uji ANOVA
anova_result = analyzer.perform_anova(music_data)
print("ANOVA result for Danceability by Genre:", anova_result)

# Distribusi Danceability
analyzer.plot_distribution(music_data, 'danceability', "Distribution of Danceability", 'blue')

# Distribusi Energy
analyzer.plot_distribution(music_data, 'energy', "Distribution of Energy", 'orange')
