import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


def calculate_prediction(k, movie, profile, sim_m):
    n = 0
    i = 0
    total = 0

    sim = sim_m.loc[movie][:]
    sim.loc[movie] = 0
    sim = sim.sort_values(ascending=False)
    while n < k and i < len(sim) - 1:
        neig = sim.index[i]
        if neig in profile.index:
            total = total + sim.iloc[i]
            n = n + 1
        i = i + 1

    return total


def generate_map(number, k, user_item_m: pd.DataFrame, sim_m: pd.DataFrame, users_v: np.ndarray, test_data_m: pd.DataFrame):
    map_users = pd.DataFrame(index=users, columns=['map'])
    map_users = map_users.sort_index()
    for u in users_v:
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
aspect_movie_data = pd.read_csv("./movie_aspects_matrix_5.csv")
aspect_movie_data.columns = aspect_movie_columns
movie_aspects_matrix = aspect_movie_data.pivot(index="movie_id", columns="aspect", values="score")
movie_aspects_matrix = movie_aspects_matrix.fillna(0)

# jaccard Sim Matrix
# jac_sim = 1 - pairwise_distances(user_item.T, metric="hamming")
# jac_sim = pd.DataFrame(jac_sim, index=user_item.columns, columns=user_item.columns)

# pearson sim matrix
# pearson_sim = movie_aspects_matrix.corr('pearson')

# cosine sim matrix
cosine_sim = cosine_similarity(movie_aspects_matrix)
cosine_sim = pd.DataFrame(cosine_sim, index=user_item.columns, columns=user_item.columns)

# read data set and
print("--- Generating Predictions and MAP ---")
test_data = pd.read_csv("../Base de Dados HetRec Arpit/test.csv", usecols=used_columns)
users = test_data.user_id.unique()
test_data.index = test_data.user_id

k_values = [2, 5, 10]
n_values = [1, 5, 10]

f = open("Results_5_Aspects/map_content_item_knn_cosine_5.txt", "w")
f.write("--- ITEM-KNN RESULTS ---\n")
print("--- ITEM-KNN RESULTS ---")
for k in k_values:
    for n in n_values:
        map_value = generate_map(n, k, user_item, cosine_sim, users, test_data)
        f.write("K = " + str(k) + " MAP@" + str(n) + " = " + str(map_value) + "\n")
        print("K = " + str(k) + " MAP@" + str(n) + " = " + str(map_value) + "\n")
f.close()
