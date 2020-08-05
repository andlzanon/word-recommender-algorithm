import numpy as np
import pandas as pd

train_data = pd.read_csv("../Base de Dados HetRec Arpit/train.csv", usecols=used_columns)

# generate user/item matrix and mean item and transform it into interactions
user_item = train_data.pivot(index="user_id", columns="movie_id", values="rating")
user_item[user_item >= 0] = 1
user_item[user_item.isna()] = 0

# return a data_frame with the number of times for each movie_id was interacted
times_seen = user_item.sum(axis=0)
times_seen = times_seen.sort_values(ascending=False)

# topn is a np.arrray with the top n (number) most watched films 
number = 10
topn = np.array(times_seen.index[:number])
