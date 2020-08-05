import numpy as np
import pandas as pd
import os
from scipy import sparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
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

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

reviews = pd.DataFrame("", index=user_item.columns, columns=['text'])
for movie_id in user_item.columns:
    with open('../Base de Dados HetRec Arpit/HetRec_Reviews/' + str(movie_id) + ".txt", encoding="utf8") as file:
        text = file.read().replace('\n', '')

    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words and not w.isnumeric() and w.isalpha():
            filtered_sentence.append(w)

    final_text = ""
    for w in filtered_sentence:
        final_text = final_text + " " + w

    reviews.at[int(movie_id)] = final_text

reviews = reviews.sort_values(by=['movie_id'])
# get vector with all sentences where index = movie_id
vec_reviews_condensed = []
for index, row in reviews.iterrows():
    vec_reviews_condensed.append(row[0])

# call TFIDF from sklearn and calculate cosine similarity
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(vec_reviews_condensed)
sparse_matrix = sparse.csr_matrix(vectors)
movies_similarities = cosine_similarity(sparse_matrix)
sim_matrix = pd.DataFrame(movies_similarities, index=user_item.columns, columns=user_item.columns)

print("end")