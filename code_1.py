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
new_df.head() 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words= 'english')
vectors = cv.fit_transform(new_df['tags']).toarray()
vectors
vectors[0]
cv.get_feature_names_out()
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y = []

    for i in text.split():
       y.append( ps.stem(i))
    return " ".join(y)
ps.stem('loved')
new_df['tags'] = new_df['tags'].apply(stem)
new_df['tags']
cv.get_feature_names_out()
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
similarity[2]
def recommend(movie):
    movie_index = new_df [new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key = lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)
recommend('Spectre')
new_df.iloc[1216].title
import pickle 
pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(new_df.to_dict(), open('movie_dict.pkl' , 'wb'))
