import numpy as np
import pandas as pd
import math

header = ['user_id', 'movie_id', 'rating']

data_set = pd.read_csv("../Base de Dados HetRec Arpit/1851_movies_ratings.txt", header=None, names=header,
sep='\t')

idf_df = pd.DataFrame(columns=['aspect', 'freq', 'idf'])
idf_df.set_index('aspect', inplace=True)

aspects_n = 100
movies = data_set.movie_id.unique()
movies.sort()
N = len(movies)
for movie in movies:
    # get top v aspects, add the movie_id column and append to all
    top_aspects = pd.read_csv("../new_aspects/movie_" + str(movie) + ".csv", usecols=['aspect', 'score'])[:aspects_n]

    for aspect in top_aspects.aspect.unique():
        if aspect in idf_df.index:
            idf_df.at[aspect] = [idf_df.loc[aspect].freq + 1, 0]
        else:
            idf_df.loc[aspect] = [1, 0]

for aspect in idf_df.index:
    dft = float(idf_df.loc[aspect].freq)
    idf = math.log(N / dft)
    idf_df.at[aspect] = [idf_df.loc[aspect].freq, idf]

idf_df = idf_df.sort_values('idf')

for movie in movies:
    movie_aspects = pd.DataFrame(columns=['aspect', 'score'])
    movie_aspects.set_index('aspect', inplace=True)
    # get top v aspects, add the movie_id column and append to all
    top_aspects = pd.read_csv("../new_aspects/movie_" + str(movie) + ".csv", usecols=['aspect', 'score'])[:aspects_n]

    for index, row in top_aspects.iterrows():
        aspect = row[0]
        score = row[1]
        if idf_df.loc[aspect].idf >= 2:
            movie_aspects.loc[aspect] = [score]

    movie_aspects.to_csv('../tf_aspects/movie_' + str(movie) + ".csv", mode='w', index=True, header=True)

print("end")
