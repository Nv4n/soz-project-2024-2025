import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint as pp
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


# This code snippet is reading data from CSV files into pandas DataFrames. Here's a breakdown of what
# each line is doing:
df_ratings = pd.read_csv("src/data/Ratings.csv", na_values=["null", "nan", ""])
df_books = pd.read_csv(
    "src/data/Books.csv",
    na_values=["null", "nan", ""],
    usecols=["ISBN", "Book-Title", "Book-Author"],
)
df_users = pd.read_csv("src/data/Users.csv", na_values=["null", "nan", ""])

df_books = df_books.fillna("NaN")
df_ratings = df_ratings.dropna()
df_users = df_users.fillna(-1)

# This code snippet is performing the following operations:
combine_book_ratings = pd.merge(df_ratings, df_books, on="ISBN")
combine_book_ratings = combine_book_ratings.drop(["Book-Author"], axis="columns")

book_rating_count = (
    combine_book_ratings.groupby(by=["Book-Title", "ISBN"])["Book-Rating"]
    .count()
    .reset_index()
    .rename(columns={"Book-Rating": "RatingCount"})
)[["ISBN", "Book-Title", "RatingCount"]]


book_rating_with_total_count = combine_book_ratings.merge(
    book_rating_count, on=["ISBN", "Book-Title"], how="left"
)

pp(book_rating_with_total_count["RatingCount"].quantile(np.arange(0.9, 1, 0.01)))
popularity_threshold = 136

rating_popular_books = book_rating_with_total_count.query(
    "RatingCount >= @popularity_threshold"
)

pivot = (
    rating_popular_books.drop_duplicates(["Book-Title", "User-ID"])
    .pivot(index="Book-Title", columns="User-ID", values="Book-Rating")
    .fillna(0)
)
# matrix_popular_books = csr_matrix(pivot.values)

model_knn = NearestNeighbors(metric="cosine", algorithm="auto")

model_knn.fit(csr_matrix(pivot.values))


# TODO LOOK AT ANN with ANNOY OR FAISS
# TODO Look
def get_recommends(book=""):
    try:
        x = pivot.loc[book].array.reshape(1, -1)
        distances, indices = model_knn.kneighbors(x, n_neighbors=5)
        R_books = []
        for distance, indice in zip(distances[0], indices[0]):
            if distance != 0:
                R_book = pivot.index[indice]
                R_books.append([R_book, distance])
        recommended_books = [book, R_books[::-1]]
        return recommended_books
    except:
        return f"{book} is not in the top books"


pp(get_recommends("Tess of the D'Urbervilles (Wordsworth Classics)"))
