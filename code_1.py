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
