import numpy as np
import pandas as pd
import rec_functions as rec_func
import nltk
from nltk.corpus import wordnet as wn
import os.path


# generate similarity between movies i and j by calling
# Wu-Palmer Similarity with wordnet
def similarity_ij(n_aspects: int, aspects_i: list, aspects_j: list):
    sum = 0
    n_tuples = 0
    for i in range(0, n_aspects):
        sim = wn.wup_similarity(aspects_i[i], aspects_j[i])
        if sim is not None:
            sum = sum + (sim * (n_aspects - i))
            n_tuples = n_tuples + (n_aspects - i)

    if n_tuples > 0:
        return sum / n_tuples
    else:
        return 0


# get n (number) aspects most important to the movie that
# has a meaning on wordnet
def get_n_aspects(number: int, aspects_movie: pd.DataFrame):
    n = 0
    index = 0
    output = []
    while n < number:
        syns = wn.synsets(str(aspects_movie.iloc[index][0]))
        if len(syns) > 0:
            output.append(syns[0])
            n = n + 1
        index = index + 1

    return output


# generate movie x movie similarity matrix by accessing its document with aspects and comparing the words with wordnet
def generate_sim_matrix(n_aspects: int, user_item_m: pd.DataFrame):
    sim_movies = pd.DataFrame(0, index=user_item_m.columns, columns=user_item_m.columns)
    n = len(user_item_m.columns)
    nltk.download('wordnet')
    columns = ['aspect', 'score']

    for i in range(0, n):
        movie_i = user_item_m.columns[i]
        filename_mi = "../new_aspects/movie_" + str(movie_i) + ".csv"
        if os.path.isfile(filename_mi):
            aspects_movie_i = pd.read_csv(filename_mi, usecols=columns)
            top_aspects_movie_i = get_n_aspects(n_aspects, aspects_movie_i)
            for j in range(i, n):
                movie_j = user_item_m.columns[j]
                filename_mj = "../new_aspects/movie_" + str(movie_j) + ".csv"
                if os.path.isfile(filename_mj):
                    aspects_movie_j = pd.read_csv(filename_mj, usecols=columns)
                    top_aspects_movie_j = get_n_aspects(n_aspects, aspects_movie_j)
                    sim_ij = similarity_ij(n_aspects, top_aspects_movie_i, top_aspects_movie_j)
                    sim_movies.loc[movie_i, movie_j] = sim_ij
                    sim_movies.loc[movie_j, movie_i] = sim_ij
                    print("sim between " + str(movie_i) + " and " + str(movie_j) + " is: " + str(sim_ij))

    return sim_movies


# if a floating point error occurs, an error is raised:
np.seterr(all='raise')

print("--- Generating User Item Matrix ---")
used_columns = ['user_id', 'movie_id', 'rating']

train_data = pd.read_csv("../Base de Dados HetRec Arpit/train.csv", usecols=used_columns)

# generate user/item matrix and mean item and transform it into interactions
user_item = train_data.pivot(index="user_id", columns="movie_id", values="rating")
user_item[user_item >= 0] = 1
user_item[user_item.isna()] = 0

print("--- Generating Similarity Matrix ---")

# these two lines below must be not commented if you want to generat the similarity matrix:
sim_matrix = generate_sim_matrix(25, user_item)
sim_matrix.to_csv("sim_matrix_25.csv", mode='w', header=False, index=False)

# semantic_sim = pd.read_csv("./sim_matrix.csv", header=None)
# semantic_sim.index = user_item.columns
# semantic_sim.columns = user_item.columns

print("--- Generating Predictions and MAP ---")

test_data = pd.read_csv("../Base de Dados HetRec Arpit/test.csv", usecols=used_columns)

users = test_data.user_id.unique()

# the rows of the dataframe are users id:
test_data.index = test_data.user_id

k_values = [2, 5, 10]
n_values = [1, 5, 10]

f = open("Final_Results/map_new2_word_rec_25.txt", "w")
f.write("--- WORD-RECOMMENDER RESULTS ---\n")
print("--- WORD-RECOMMENDER RESULTS ---")
for k in k_values:
    for n in n_values:
        # 'users' is not sorted
        map_value = rec_func.generate_map(n, k, user_item, sim_matrix, users, test_data)
        f.write("K = " + str(k) + " MAP@" + str(n) + " = " + str(map_value) + "\n")
        print("K = " + str(k) + " MAP@" + str(n) + " = " + str(map_value) + "\n")
f.close()
