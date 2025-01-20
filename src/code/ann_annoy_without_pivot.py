from annoy import AnnoyIndex
import pandas as pd
import numpy as np
from collections import defaultdict
from pprint import pprint as pp
from scipy.spatial.distance import cosine

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
    combine_book_ratings.groupby("ISBN")["Book-Rating"]
    .count()
    .reset_index()
    .rename(columns={"Book-Rating": "RatingCount"})
)
combine_book_ratings = combine_book_ratings.merge(
    book_rating_count, on="ISBN", how="left"
)

# Thresholds
popularity_threshold = 136
popular_books = combine_book_ratings.query("RatingCount >= @popularity_threshold")
unpopular_books = combine_book_ratings.query("RatingCount < @popularity_threshold")

# Most popular books fallback
most_popular_books = (
    popular_books.groupby("ISBN")
    .size()
    .sort_values(ascending=False)
    .index[:10]
    .tolist()
)

# Map users and books to indices
user_to_index = {
    user: idx for idx, user in enumerate(combine_book_ratings["User-ID"].unique())
}
book_to_index = {
    isbn: idx for idx, isbn in enumerate(combine_book_ratings["ISBN"].unique())
}
index_to_book = {idx: isbn for isbn, idx in book_to_index.items()}

# Create user-book ratings dictionary
book_vectors = defaultdict(lambda: np.zeros(len(user_to_index)))
for _, row in combine_book_ratings.iterrows():
    book_idx = book_to_index[row["ISBN"]]
    user_idx = user_to_index[row["User-ID"]]
    book_vectors[book_idx][user_idx] = row["Book-Rating"]

# Build Annoy index
num_users = len(user_to_index)
annoy_index = AnnoyIndex(num_users, metric="angular")

for book_idx, vector in book_vectors.items():
    annoy_index.add_item(book_idx, vector)

annoy_index.build(n_trees=10)


# Recommendation function
def get_recommends(isbn="", k_neighbors=5):
    try:
        if isbn in book_to_index:
            book_idx = book_to_index[isbn]
            nearest_neighbors = annoy_index.get_nns_by_item(
                book_idx, k_neighbors, include_distances=True
            )
            R_books = []
            for neighbor_idx, distance in zip(*nearest_neighbors):
                neighbor_isbn = index_to_book[neighbor_idx]
                R_book = combine_book_ratings[
                    combine_book_ratings["ISBN"] == neighbor_isbn
                ]["Book-Title"].values[0]
                R_books.append([R_book, distance])
            return [isbn, R_books[::-1]]
        elif isbn in unpopular_books["ISBN"].values:
            # Dynamically create vector for unpopular book
            isbn_vector = np.zeros(num_users)
            book_data = combine_book_ratings[combine_book_ratings["ISBN"] == isbn]
            for _, row in book_data.iterrows():
                user_idx = user_to_index[row["User-ID"]]
                isbn_vector[user_idx] = row["Book-Rating"]

            nearest_neighbors = annoy_index.get_nns_by_vector(
                isbn_vector, k_neighbors, include_distances=True
            )
            R_books = []
            for neighbor_idx, distance in zip(*nearest_neighbors):
                neighbor_isbn = index_to_book[neighbor_idx]
                R_book = combine_book_ratings[
                    combine_book_ratings["ISBN"] == neighbor_isbn
                ]["Book-Title"].values[0]
                R_books.append([R_book, distance])
            return [isbn, R_books[::-1]]
        else:
            # Fallback to most popular books
            R_books = [
                [
                    combine_book_ratings[combine_book_ratings["ISBN"] == book][
                        "Book-Title"
                    ].values[0],
                    0,
                ]
                for book in most_popular_books[:k_neighbors]
            ]
            return [isbn, R_books]
    except Exception as e:
        return str(e)


# Test the function
pp(get_recommends("1558745157"))


pp(get_recommends("0330281747"))
