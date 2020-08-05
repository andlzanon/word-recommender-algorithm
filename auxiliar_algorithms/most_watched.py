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

# get data of all films and set index as movie id
movie_data = pd.read_csv("../Base de Dados HetRec Arpit/Items - hetrec_after_2000.dat", header=None, names=['movie_id', 'imdb_id', 'title', 'year', 'url'],
sep='\t')
movie_data.index = movie_data.movie_id

# create imbs_id_vec that contains the ids of the most watched films
# if imbd_id has len of 6 than add a zero to the start since imdb_id
# has tt + 7 digits
imdb_id_vec = []
for m in topn:
    m_data = movie_data.loc[m]
    m_imbd = str(m_data['imdb_id'])
    if len(m_imbd) < 7:
        imbd_id_str = "tt0" + m_imbd
    else:
        imbd_id_str = "tt" + m_imbd
        
    imdb_id_vec.append(imbd_id_str)

