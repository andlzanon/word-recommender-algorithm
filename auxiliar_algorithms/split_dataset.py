import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

header = ['user_id', 'movie_id', 'rating']

data_set = pd.read_csv("../Base de Dados HetRec Arpit/1851_movies_ratings.txt", header=None, names=header,
sep='\t')

data_set = data_set.drop_duplicates(subset =['user_id', 'movie_id'], keep = 'last')

# split data set into train and test
train, test = train_test_split(data_set, test_size=0.1, random_state=42)

train.to_csv('train.csv', mode='w', header=header, index=False)  
test.to_csv('test.csv', mode='w', header=header, index=False)  
