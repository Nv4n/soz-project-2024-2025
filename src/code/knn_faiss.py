import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint as pp

# Load datasets
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

# Merge ratings with book metadata
combine_book_ratings = pd.merge(df_ratings, df_books, on="ISBN")
combine_book_ratings = combine_book_ratings.drop(["Book-Author"], axis="columns")

# Calculate popularity threshold
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
unpopular_books = book_rating_with_total_count.query(
    "RatingCount < @popularity_threshold"
)

# Create embeddings for popular books
pivot_popular = (
    popular_books.drop_duplicates(["ISBN", "User-ID"])
    .pivot(index="ISBN", columns="User-ID", values="Book-Rating")
    .fillna(0)
)

# Normalize vectors for cosine similarity
popular_embeddings = normalize(pivot_popular.values, axis=1, norm="l2")
isbn_list_popular = pivot_popular.index.tolist()

# Index popular book embeddings with Faiss
d = popular_embeddings.shape[1]
index = faiss.IndexFlatIP(d)  # Cosine similarity
index.add(popular_embeddings)

# Handle unpopular books using TF-IDF on book titles
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
unpopular_titles = df_books[df_books["ISBN"].isin(unpopular_books["ISBN"])]
tfidf_embeddings = vectorizer.fit_transform(unpopular_titles["Book-Title"]).toarray()

isbn_list_unpopular = unpopular_titles["ISBN"].tolist()
index_unpopular = faiss.IndexFlatIP(tfidf_embeddings.shape[1])
index_unpopular.add(normalize(tfidf_embeddings, axis=1, norm="l2"))


# Recommendation function
def get_recommends(isbn, k_neighbors=5):
    if isbn in isbn_list_popular:
        idx = isbn_list_popular.index(isbn)
        distances, indices = index.search(
            popular_embeddings[idx].reshape(1, -1), k_neighbors
        )
        recommended_books = [
            [isbn_list_popular[i], distances[0][j]]
            for j, i in enumerate(indices[0])
            if distances[0][j] > 0
        ]
    elif isbn in isbn_list_unpopular:
        idx = isbn_list_unpopular.index(isbn)
        distances, indices = index_unpopular.search(
            normalize(tfidf_embeddings[idx].reshape(1, -1)), k_neighbors
        )
        recommended_books = [
            [isbn_list_unpopular[i], distances[0][j]]
            for j, i in enumerate(indices[0])
            if distances[0][j] > 0
        ]
    else:
        return f"{isbn} is not found in the dataset."

    return [isbn, recommended_books]


# Test the function
pp(get_recommends("1558745157"))
pp(get_recommends("0330281747"))
