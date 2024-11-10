import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

def describe_data(df, feature_columns):
    """Mengembalikan deskripsi statistik dari data."""
    return df[feature_columns].describe()

def calculate_median(df, feature_columns):
    """Mengembalikan median dari setiap fitur numerik."""
    return df[feature_columns].median()

def plot_correlation_matrix(df, feature_columns):
    """Membuat heatmap korelasi antar fitur numerik."""
    correlation_matrix = df[feature_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")
    plt.title("Correlation Matrix of Audio Features")
    plt.show()

def plot_danceability_by_genre(df):
    """Membuat boxplot danceability berdasarkan genre."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='genre', y='danceability')
    plt.title("Danceability by Genre")
    plt.xticks(rotation=90)
    plt.show()

def perform_anova(df):
    """Melakukan uji ANOVA untuk danceability berdasarkan genre."""
    genres = df['genre'].unique()
    danceability_data = [df[df['genre'] == genre]['danceability'] for genre in genres]
    return f_oneway(*danceability_data)

def plot_distribution(df, column, title, color):
    """Membuat histogram distribusi untuk kolom tertentu."""
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], bins=20, kde=True, color=color)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()
