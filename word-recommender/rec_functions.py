import numpy as np
import pandas as pd

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
    sim = sim.sort_values(ascending=False)
    while n < k and i < len(sim) - 1:
        neig = sim.index[i]
        if neig in profile.index:
            total = total + sim.iloc[i]
            n = n + 1
        i = i + 1

    return total


def generate_map(number, k, user_item_m: pd.DataFrame, sim_m: pd.DataFrame, users_v: np.array, test_data_m: pd.DataFrame):
    map_users = pd.DataFrame(index=users_v, columns=['map'])
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
        len_rel = len(relevants)
        n_hits = 0
        ap = 0
        for rank in range(0, number):
            top_m = pred_at_n.index[rank]
            if top_m in np.array(relevants.movie_id):
                n_hits = n_hits + 1
                ap = ap + (n_hits / (rank + 1))

        if n_hits > 0:
            u_ap = ap / len_rel
            map_users.loc[u] = u_ap
            print("user: " + str(u) + " AP: " + str(u_ap))
        else:
            print("user: " + str(u) + " AP: 0")
            map_users.loc[u] = 0

    return map_users.mean()['map']