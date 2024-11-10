# Mengecek nilai yang hilang
missing_values = music_data.isnull().sum()
print("Missing values per column:\n", missing_values[missing_values > 0])

# Menghapus atau mengisi data yang hilang jika diperlukan
music_data = music_data.dropna()  # Menghapus baris yang memiliki nilai NaN

# Mengecek duplikasi
duplicates = music_data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
music_data = music_data.drop_duplicates()  # Menghapus duplikasi data jika ada
