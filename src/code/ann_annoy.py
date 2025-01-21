from annoy import AnnoyIndex
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pprint import pprint as pp
from sklearn.preprocessing import normalize

# Load data
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

combine_book_ratings = pd.merge(df_ratings, df_books, on="ISBN")
combine_book_ratings = combine_book_ratings.drop(["Book-Author"], axis="columns")

# Compute book popularity
book_rating_count = (
    combine_book_ratings.groupby(by=["ISBN"])["Book-Rating"]
    .count()
    .reset_index()
    .rename(columns={"Book-Rating": "RatingCount"})
)[["ISBN", "RatingCount"]]

book_rating_with_total_count = combine_book_ratings.merge(
    book_rating_count, on=["ISBN"], how="left"
)

# Thresholds

pp(book_rating_with_total_count["RatingCount"].quantile(np.arange(0.9, 1, 0.01)))
popularity_threshold = 136

popular_books = book_rating_with_total_count.query(
    "RatingCount >= @popularity_threshold"
)

# Popular annoy

# Sparse matrix
user_book_matrix = (
    popular_books.drop_duplicates(["ISBN", "User-ID"])
    .pivot(index="ISBN", columns="User-ID", values="Book-Rating")
    .fillna(0)
)

sparse_matrix = csr_matrix(user_book_matrix.values)

# Build Annoy Index for all books
num_features = sparse_matrix.shape[1]
annoy_index = AnnoyIndex(num_features, metric="angular")

isbn_to_index = {isbn: idx for idx, isbn in enumerate(user_book_matrix.index)}
index_to_isbn = {idx: isbn for isbn, idx in isbn_to_index.items()}

for idx, row in enumerate(sparse_matrix):
    annoy_index.add_item(idx, row.toarray()[0])

annoy_index.build(n_trees=10)

# Fallback to k most popular books
most_popular_books = (
    popular_books.groupby("ISBN")
    .size()
    .sort_values(ascending=False)
    .index[:10]
    .tolist()
)


# Recommendation function
def get_recommends(isbn="", k_neighbors=5):
    try:
        if isbn in isbn_to_index:
            isbn_idx = isbn_to_index[isbn]
            nearest_neighbors = annoy_index.get_nns_by_item(
                isbn_idx, k_neighbors + 1, include_distances=True
            )
            R_books = []
            for neighbor_idx, distance in zip(*nearest_neighbors):
                neighbor_isbn = index_to_isbn[neighbor_idx]
                if neighbor_isbn == isbn:  # Skip the same ISBN
                    continue
                R_book = [
                    combine_book_ratings[combine_book_ratings["ISBN"] == neighbor_isbn][
                        "Book-Title"
                    ].values[0],
                    neighbor_isbn,
                ]
                R_books.append([R_book, distance])
            return [isbn, R_books[::-1]]
        else:
            # Fallback to most popular books
            R_books = [
                [
                    [
                        combine_book_ratings[combine_book_ratings["ISBN"] == book][
                            "Book-Title"
                        ].values[0],
                        book,
                    ],
                    0,
                ]
                for book in most_popular_books[:k_neighbors]
            ]
            return [isbn, R_books]
    except Exception as e:
        pp("ERROR")
        return str(e.with_traceback())


# Test the function
pp(get_recommends("1558745157"))
pp(get_recommends("0330281747"))
