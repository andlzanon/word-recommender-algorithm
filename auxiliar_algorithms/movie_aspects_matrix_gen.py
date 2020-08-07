import numpy as np
import pandas as pd

# if a floating point error occurs, an error is raised:
np.seterr(all='raise')

used_columns = ['user_id', 'movie_id', 'rating']

train_data = pd.read_csv("../Base de Dados HetRec Arpit/train.csv", usecols=used_columns)

# generate user/item matrix and mean item and transform it into interactions
user_item = train_data.pivot(index="user_id", columns="movie_id", values="rating")

# create a new data frame with all top v aspects of all movies
all_movie_aspects = pd.DataFrame(columns=['aspect', 'score', 'movie_id'])
aspects_n = 25
n = len(user_item.columns)
for i in range(0, n):
    movie = user_item.columns[i]
    # get top v aspects, add the movie_id column and append to all
    top_v_aspects = pd.read_csv("../aspects/movie_" + str(movie) + ".csv", usecols=['aspect', 'score']).sort_values('score', ascending=False)[:aspects_n]
    top_v_aspects = top_v_aspects.assign(movie_id=[movie]*aspects_n)
    all_movie_aspects = all_movie_aspects.append(top_v_aspects)

# transform into matrix and than to database
movie_aspects_matrix = all_movie_aspects.pivot(index="movie_id", columns="aspect", values="score")
# movie_aspects_matrix = movie_aspects_matrix.fillna(0)
all_movie_aspects.to_csv('../word-recommender/movie_aspects_matrix.csv', mode='w', index=False, header=False,
                         columns=all_movie_aspects.columns.to_list())