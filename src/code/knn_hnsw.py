import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import hnswlib  # Library for HNSW
from pprint import pprint as pp

# Load data
df_ratings = pd.read_csv("src/data/Ratings.csv", na_values=["null", "nan", ""])
df_books = pd.read_csv(
    "src/data/Books.csv",
    na_values=["null", "nan", ""],
    usecols=["ISBN", "Book-Title", "Book-Author"],
)
df_users = pd.read_csv("src/data/Users.csv", na_values=["null", "nan", ""])

# Data preprocessing
df_books = df_books.fillna("NaN")
df_ratings = df_ratings.dropna()
df_users = df_users.fillna(-1)

combine_book_ratings = pd.merge(df_ratings, df_books, on="ISBN")
combine_book_ratings = combine_book_ratings.drop(["Book-Author"], axis="columns")

# Calculate book rating counts
book_rating_count = (
    combine_book_ratings.groupby("ISBN")["Book-Rating"]
    .count()
    .reset_index()
    .rename(columns={"Book-Rating": "RatingCount"})
)

# Merge rating counts with the ratings data
book_rating_with_total_count = combine_book_ratings.merge(
    book_rating_count, on="ISBN", how="left"
)

# Filter by popularity threshold
popularity_threshold = 136
rating_popular_books = book_rating_with_total_count.query(
    "RatingCount >= @popularity_threshold"
).copy()

# Filter out inactive users
user_activity = rating_popular_books.groupby("User-ID")["Book-Rating"].count()
active_users = user_activity[user_activity >= 5].index
rating_popular_books = rating_popular_books[rating_popular_books["User-ID"].isin(active_users)]

# Precompute mappings for users and books
user_id_mapping = {user_id: idx for idx, user_id in enumerate(rating_popular_books["User-ID"].unique())}
isbn_mapping = {isbn: idx for idx, isbn in enumerate(rating_popular_books["ISBN"].unique())}

# Map User-ID and ISBN to respective indices
rating_popular_books["User-Idx"] = rating_popular_books["User-ID"].map(user_id_mapping)
rating_popular_books["ISBN-Idx"] = rating_popular_books["ISBN"].map(isbn_mapping)

# Create sparse matrix
sparse_matrix = coo_matrix(
    (rating_popular_books["Book-Rating"],
     (rating_popular_books["ISBN-Idx"], rating_popular_books["User-Idx"])),
    shape=(len(isbn_mapping), len(user_id_mapping))
).tocsr()

# Reverse mappings for later use
index_to_isbn = {idx: isbn for isbn, idx in isbn_mapping.items()}
index_to_user_id = {idx: user_id for user_id, idx in user_id_mapping.items()}

# Initialize HNSW index using hnswlib
dim = sparse_matrix.shape[1]  # Number of features (users)
index = hnswlib.Index(space='cosine', dim=dim)  # Use cosine distance for similarity
index.init_index(max_elements=sparse_matrix.shape[0], ef_construction=200, M=16)

# Add data to the index
index.add_items(sparse_matrix.toarray())

# Set efSearch for querying (larger value makes the search more accurate but slower)
index.set_ef(50)

# Recommendation function
def get_recommends(isbn, k_neighbors=5):
    try:
        # Validate ISBN
        if isbn not in isbn_mapping:
            return f"{isbn} is not in the top books"

        # Get query vector
        isbn_idx = isbn_mapping[isbn]
        query_vector = sparse_matrix[isbn_idx].toarray().flatten()

        # Query the HNSW index
        indices, distances = index.knn_query(query_vector, k=k_neighbors)

        # Generate recommendations
        recommendations = []
        for distance, indice in zip(distances[0], indices[0]):
            if distance < 1:  # Filter for meaningful results
                recommended_isbn = index_to_isbn[indice]
                book_title = combine_book_ratings[
                    combine_book_ratings["ISBN"] == recommended_isbn
                ]["Book-Title"].values[0]
                recommendations.append([book_title, distance])

        return [isbn, recommendations[::-1]]
    except Exception as e:
        return str(e)

# Test example
example_isbn = "1558745157"  # Replace with an ISBN from your dataset
pp(get_recommends(example_isbn, k_neighbors=5))