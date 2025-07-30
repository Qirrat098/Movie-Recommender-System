import pandas as pd
import numpy as np
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
print(movies.shape)

#genres, id, keywords, title, overview, cast, crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print(movies.head(1))