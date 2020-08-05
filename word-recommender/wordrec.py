import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
import os.path

'''
def calculate_prediction:
    This function computes the estipulated prediction described
    in formula 3 of the present article; given a user 'u', 
    and a item 'i', not already ranked by the user, this function
    computes a prediction value about how the user 'u', after 
    interacting with item 'i', would have ranked the item 'i'.

    Given the k-most similar items to 'i', the metrics involved to
    generate this prediciton consists of the sum of the similarity
    between item 'i' and a item 'j'; 'j' is defined as an item 
    included in the'k-most similar' set to 'i', and 'j' must have
    been already ranked by the user 'u'.
    
def generate_map
    This function computes the MAP for every unique user_id on the test set.
    Initially it is obtained the row of the user on the user_item matrix and 
    separated into the films he has seen (profile) and not seen (prediction).
    Than for every movie on the prediction set is calculated the value of the 
    prediction in order to sort by highest to the lowest. The top N is returned
    with a split on the vector
    
    AP calculation: Than, for every movie on the TOP N is checked if it is on
    the test set if the user actually have seen. AP is the sum for evey position
    1, 2, 3, ..., n of local_hits/local_total divided by the total number of hits 
    
    the MAP is the average of all AP of the users
'''


def calculate_prediction(k, movie, profile, sim_m):
    n = 0
    i = 0
    total = 0

    sim = sim_m.loc[movie][:]
    sim.loc[movie] = 0

    # the dataframe 'sim' is sorted in descending order (from highest to lowest),
    # since is desired to iterate untill the k-highest similarities
    sim = sim.sort_values(ascending=False)

    while n < k and i < len(sim) - 1:
        neig = sim.index[i]

        # if 'neig' (the current movie being iterated) has already been seen by
        # the user, it's similarity is accumulated on 'total':

        if neig in profile.index:
            total = total + sim.iloc[i]
            n = n + 1
        i = i + 1

    return total


def generate_map(number: int, k: int, user_item_m: pd.DataFrame, sim_m: pd.DataFrame, users_v: np.ndarray,
                 test_data_m: pd.DataFrame):
    map_users = pd.DataFrame(index=users, columns=['map'])
    for u in users_v:
        # only the row containing the u-th user is selected:
        u_row = user_item_m.loc[u][:]
        profile = u_row[u_row == 1]
        prediction = u_row[u_row == 0]
        for m in prediction.index:
            prediction.loc[m] = calculate_prediction(k, m, profile, sim_m)

        prediction = prediction.sort_values(ascending=False)
        pred_at_n = prediction[:number]
        relevants = test_data_m.loc[u]
        n_hits = 0
        ap = 0
        for rank in range(0, number):
            top_m = pred_at_n.index[rank]
            if top_m in np.array(relevants.movie_id):
                n_hits = n_hits + 1
                ap = ap + (n_hits / (rank + 1))

        if n_hits > 0:
            u_ap = ap / n_hits
            map_users.loc[u] = u_ap
            print("user: " + str(u) + " AP: " + str(u_ap))
        else:
            print("user: " + str(u) + " AP: 0")
            map_users.loc[u] = 0

    return map_users.mean()['map']


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
        filename_mi = "../aspects/movie_" + str(movie_i) + ".csv"
        if os.path.isfile(filename_mi):
            aspects_movie_i = pd.read_csv(filename_mi, usecols=columns).sort_values('score', ascending=False)
            top_aspects_movie_i = get_n_aspects(n_aspects, aspects_movie_i)
            for j in range(i, n):
                movie_j = user_item_m.columns[j]
                filename_mj = "../aspects/movie_" + str(movie_j) + ".csv"
                if os.path.isfile(filename_mj):
                    aspects_movie_j = pd.read_csv(filename_mj, usecols=columns).sort_values('score', ascending=False)
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
# sim_matrix = generate_sim_matrix(5, user_item)
# sim_matrix.to_csv("sim_matrix.csv", mode='w', header=False, index=False)

semantic_sim = pd.read_csv("./sim_matrix.csv", header=None)
semantic_sim.index = user_item.columns
semantic_sim.columns = user_item.columns

print("--- Generating Predictions and MAP ---")

test_data = pd.read_csv("../Base de Dados HetRec Arpit/test.csv", usecols=used_columns)

users = test_data.user_id.unique()
users = users.sort()

# the rows of the dataframe are users id:
test_data.index = test_data.user_id

k_values = [2, 5, 10, 15]
n_values = [10]

f = open("map_word_rec.txt", "w")
f.write("--- WORD-RECOMMENDER RESULTS ---\n")
print("--- WORD-RECOMMENDER RESULTS ---")
for k in k_values:
    for n in n_values:
        # 'users' is not sorted
        map_value = generate_map(n, k, user_item, semantic_sim, users, test_data)
        f.write("K = " + str(k) + " MAP@" + str(n) + " = " + str(map_value) + "\n")
        print("K = " + str(k) + " MAP@" + str(n) + " = " + str(map_value) + "\n")
f.close()

# with open("map_word_rec.txt", 'a+', encoding="utf-8") as f:
#    print("--- WORD-RECOMMENDER RESULTS ---\n", file=f)
#    for k in k_values:
#        for n in n_values:
#            map_value = generate_map(n, k, user_item, semantic_sim, users, test_data)
#            print("K = " + str(k) + " MAP@" + str(n) + " = " + str(map_value) + "\n", file=f)
#           print("K = " + str(k) + " MAP@" + str(n) + " = " + str(map_value) + "\n", file=f)
