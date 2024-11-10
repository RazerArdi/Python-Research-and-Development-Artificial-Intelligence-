import pandas as pd

def load_data(path):
    """Membaca dataset dari path yang diberikan."""
    return pd.read_csv(path)

def clean_data(df):
    """Membersihkan dataset dari nilai hilang dan duplikasi."""
    df = df.dropna()  # Menghapus baris dengan nilai NaN
    df = df.drop_duplicates()  # Menghapus duplikasi
    return df

def get_missing_values(df):
    """Mengecek nilai yang hilang."""
    missing_values = df.isnull().sum()
    return missing_values[missing_values > 0]

def get_feature_columns():
    """Mengembalikan daftar kolom fitur numerik yang digunakan untuk rekomendasi."""
    return ['danceability', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'energy']
