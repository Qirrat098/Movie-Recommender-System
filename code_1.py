import pandas as pd
import numpy as np
import ast

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
print(movies.shape)

#genres, id, keywords, title, overview, cast, crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print(movies.head(1))
movies.isnull().sum()
movies.dropna(inplace = True)
movies.duplicated().sum()
movies[movies.duplicated()]
movies = movies.drop_duplicates()
movies.duplicated().sum()
movies.iloc[0].genres
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies['genres'].apply(convert)
movies.head()
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()
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
movies["crew"][0]
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
movies['crew'] = movies['crew'].apply(fetch_director)
movies.head()
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.head()
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])
movies.head()
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head()
new_df = movies[['movie_id', 'title', 'tags']]
print(new_df.head())
new_df['tags'] = new_df['tags'].apply(lambda x:"".join(x))
new_df['tags'][2]
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())