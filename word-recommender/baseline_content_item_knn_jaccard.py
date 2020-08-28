import numpy as np
import pandas as pd
import rec_functions as rec_func
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

# read dataset
np.seterr(all='raise')
used_columns = ['user_id', 'movie_id', 'rating']
train_data = pd.read_csv("../Base de Dados HetRec Arpit/train.csv", usecols=used_columns)

# generate user/item matrix and mean item and transform it into interactions
print("--- Generating User Item Matrix ---")
user_item = train_data.pivot(index="user_id", columns="movie_id", values="rating")
user_item[user_item >= 0] = 1
user_item[user_item.isna()] = 0

# generate similarity matrix
print("--- Generating Similarity Matrix ---")

# get movie aspect matrix and fill it with 0 instead of nan a
aspect_movie_columns = ['aspect', 'score', 'movie_id']
aspect_movie_data = pd.read_csv("movie_aspects_matrix_5.csv")
aspect_movie_data.columns = aspect_movie_columns
movie_aspects_matrix = aspect_movie_data.pivot(index="movie_id", columns="aspect", values="score")
movie_aspects_matrix[movie_aspects_matrix >= 0] = 1
movie_aspects_matrix = movie_aspects_matrix.fillna(0)

# jaccard Sim Matrix
jac_sim = 1 - pairwise_distances(movie_aspects_matrix, metric="hamming")
jac_sim = pd.DataFrame(jac_sim, index=movie_aspects_matrix.index, columns=movie_aspects_matrix.index)

# pearson sim matrix
# pearson_sim = movie_aspects_matrix.corr('pearson')

# cosine sim matrix
# cosine_sim = cosine_similarity(user_item)

# read data set and
print("--- Generating Predictions and MAP ---")
test_data = pd.read_csv("../Base de Dados HetRec Arpit/test.csv", usecols=used_columns)
users = test_data.user_id.unique()
test_data.index = test_data.user_id

k_values = [2, 5, 10]
n_values = [1, 5, 10]

f = open("Final_Results/final_map_content_item_knn_jaccard_5.txt", "w")
f.write("--- ITEM-KNN RESULTS ---\n")
print("--- ITEM-KNN RESULTS ---")
for k in k_values:
    for n in n_values:
        map_value = rec_func.generate_map(n, k, user_item, jac_sim, users, test_data)
        f.write("K = " + str(k) + " MAP@" + str(n) + " = " + str(map_value) + "\n")
        print("K = " + str(k) + " MAP@" + str(n) + " = " + str(map_value) + "\n")
f.close()
