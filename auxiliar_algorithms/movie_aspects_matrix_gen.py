import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# if a floating point error occurs, an error is raised:
np.seterr(all='raise')

header_data_set = ['user_id', 'movie_id', 'rating']
data_set = pd.read_csv("../Base de Dados HetRec Arpit/1851_movies_ratings.txt", header=None, names=header_data_set,
sep='\t')
movies = data_set.movie_id.unique()
movies.sort()

# create a new data frame with all top v aspects of all movies
all_movie_aspects = pd.DataFrame(columns=['aspect', 'score', 'movie_id'])
aspects_n = 5
n = len(movies)
for i in range(0, n):
    movie = movies[i]
    # get top v aspects, add the movie_id column and append to all
    top_v_aspects = pd.read_csv("../new_aspects/movie_" + str(movie) + ".csv", usecols=['aspect', 'score'])[:aspects_n]
    top_v_aspects = top_v_aspects.assign(movie_id=[movie]*aspects_n)
    all_movie_aspects = all_movie_aspects.append(top_v_aspects)

# transform into matrix and than to database
movie_aspects_matrix = all_movie_aspects.pivot(index="movie_id", columns="aspect", values="score")
movie_aspects_matrix = movie_aspects_matrix.fillna(0)
all_movie_aspects.to_csv('../word-recommender/movie_aspects_matrix_5.csv', mode='w', index=False, header=False,
                         columns=all_movie_aspects.columns.to_list())

# movie_aspects_matrix = movie_aspects_matrix.fillna(0)
# cosine_sim = cosine_similarity(movie_aspects_matrix)
# cosine_sim = pd.DataFrame(cosine_sim, index=movies, columns=movies)
# cosine_sim.to_csv('tf_cosine_sim_matrix_5.csv', index=False, columns=None)

print("end")