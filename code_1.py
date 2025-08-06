import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import pickle

# Load datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on title
movies = movies.merge(credits, on='title')
print("Movies shape:", movies.shape)

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
print("First movie record:\n", movies.head(1))

# Drop null values and duplicates
movies.dropna(inplace=True)
movies = movies.drop_duplicates()

# Helper function to convert stringified JSON columns to list of names
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Process genres and keywords columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Function to extract up to 3 names from the cast column
def convert2(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies["cast"] = movies["cast"].apply(convert2)

# Function to extract the director's name from the crew column
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Tokenize overview by splitting text
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces from each element in list columns
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create tags by combining overview, genres, keywords, cast, and crew
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new DataFrame with selected columns
new_df = movies[['movie_id', 'title', 'tags']]
print("New DataFrame preview:\n", new_df.head())

# Convert the tags list to a single string for each movie and normalize to lower case
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
print("Tags sample (before stemming):", new_df['tags'].iloc[2])

# Apply stemming to tags
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
print("Tags sample (after stemming):", new_df['tags'].iloc[2])

# Recompute feature vectors using CountVectorizer on stemmed tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).array() # type: ignore

# Compute cosine similarity matrix between movies
similarity = cosine_similarity(vectors)
#print("Cosine similarity sample:", similarity[2])  # Uncomment if needed

# Function to recommend movies based on cosine similarity
def recommend(movie):
    if movie not in new_df['title'].values:
        print(f"Movie '{movie}' not found in the dataset!")
        return
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    print("Recommended movies:")
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Test the recommendation function
recommend('Spectre')
print("Movie at index 1216:", new_df.iloc[1216].title)

# Save the processed DataFrame and its dictionary representation using pickle
with open('movies.pkl', 'wb') as f:
    pickle.dump(new_df, f)

with open('movie_dict.pkl', 'wb') as f:
    pickle.dump(new_df.to_dict(), f)  
    