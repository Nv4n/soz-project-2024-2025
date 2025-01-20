import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from pprint import pprint as pp

# Load data
df_ratings = pd.read_csv("src/data/Ratings.csv", na_values=["null", "nan", ""])
df_books = pd.read_csv(
    "src/data/Books.csv",
    na_values=["null", "nan", ""],
    usecols=["ISBN", "Book-Title", "Book-Author"],
)
df_users = pd.read_csv("src/data/Users.csv", na_values=["null", "nan", ""])

# Clean data
df_books = df_books.fillna("NaN")
df_ratings = df_ratings.dropna()
df_users = df_users.fillna(-1)

combine_book_ratings = pd.merge(df_ratings, df_books, on="ISBN")
combine_book_ratings = combine_book_ratings.drop(["Book-Author"], axis="columns")

book_rating_count = (
    combine_book_ratings.groupby(by=["ISBN"])["Book-Rating"]
    .count()
    .reset_index()
    .rename(columns={"Book-Rating": "RatingCount"})
)

book_rating_with_total_count = combine_book_ratings.merge(
    book_rating_count, on=["ISBN"], how="left"
)

# Threshold for popularity
popularity_threshold = 136
popular_books = book_rating_with_total_count.query("RatingCount >= @popularity_threshold")
unpopular_books = book_rating_with_total_count.query("RatingCount < @popularity_threshold")

# Prepare vectors for Faiss
def prepare_faiss_data(data):
    user_book_matrix = (
        data.drop_duplicates(["Book-Title", "User-ID"])
        .pivot(index="ISBN", columns="User-ID", values="Book-Rating")
        .fillna(0)
    )
    # Normalize the data for cosine similarity
    matrix = csr_matrix(user_book_matrix.values)
    dense_matrix = matrix.toarray()
    normalized_data = normalize(dense_matrix, axis=1)
    return normalized_data, user_book_matrix.index

# Prepare data for popular books
popular_vectors, popular_isbns = prepare_faiss_data(popular_books)

# Create Faiss index for popular books
d = popular_vectors.shape[1]
gpu_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(d))  # Cosine similarity
gpu_index.add(popular_vectors)

# Prepare embeddings for unpopular books
unpopular_vectors, unpopular_isbns = prepare_faiss_data(unpopular_books)
unpopular_index = faiss.IndexFlatIP(unpopular_vectors.shape[1])
unpopular_index.add(unpopular_vectors)

# Recommendation function
def get_recommends_faiss(isbn, k_neighbors=5):
    try:
        if isbn in popular_isbns:
            idx = np.where(popular_isbns == isbn)[0][0]
            vector = popular_vectors[idx].reshape(1, -1)
            distances, indices = gpu_index.search(vector, k_neighbors)
            recommendations = [
                [popular_isbns[i], distances[0][j]]
                for j, i in enumerate(indices[0]) if distances[0][j] > 0
            ]
        elif isbn in unpopular_isbns:
            idx = np.where(unpopular_isbns == isbn)[0][0]
            vector = unpopular_vectors[idx].reshape(1, -1)
            distances, indices = unpopular_index.search(vector, k_neighbors)
            recommendations = [
                [unpopular_isbns[i], distances[0][j]]
                for j, i in enumerate(indices[0]) if distances[0][j] > 0
            ]
        else:
            return f"{isbn} not found in the dataset."
        return [isbn, recommendations[::-1]]
    except Exception as e:
        return str(e)

# Example recommendation
pp(get_recommends_faiss("1558745157"))
