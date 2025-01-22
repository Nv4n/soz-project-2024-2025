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
    "RatingCount < @popularity_threshold & RatingCount > 0"
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
vectorizer = TfidfVectorizer(max_features=2000, lowercase=True)
unpopular_titles = df_books[df_books["ISBN"].isin(unpopular_books["ISBN"])]
unpopular_titles.loc[unpopular_titles["Book-Title"].isna(), "Book-Title"] = (
    "Unknown Title"
)

tfidf_embeddings = vectorizer.fit_transform(unpopular_titles["Book-Title"]).toarray()
isbn_list_unpopular = unpopular_titles["ISBN"].tolist()
if tfidf_embeddings.shape[0] > 0:
    tfidf_embeddings = normalize(tfidf_embeddings, axis=1, norm="l2")

pp(f"TF-IDF embeddings shape: {tfidf_embeddings.shape}")
pp(f"Zero embeddings count: {(tfidf_embeddings.sum(axis=1) == 0).sum()}")

valid_rows = tfidf_embeddings.sum(axis=1) > 0
tfidf_embeddings = tfidf_embeddings[valid_rows]
isbn_list_unpopular = [
    isbn for valid, isbn in zip(valid_rows, isbn_list_unpopular) if valid
]

pp(f"TF-IDF embeddings shape: {tfidf_embeddings.shape}")
pp(f"Zero embeddings count: {(tfidf_embeddings.sum(axis=1) == 0).sum()}")

index_unpopular = faiss.IndexFlatIP(tfidf_embeddings.shape[1])
index_unpopular.add(tfidf_embeddings)


# Fallback to k most popular books
most_popular_books = (
    popular_books.groupby("ISBN")
    .size()
    .sort_values(ascending=False)
    .index[:10]
    .tolist()
)


# Recommendation function
def get_recommends(isbn, k_neighbors=5):
    try:
        if isbn in isbn_list_popular:
            idx = isbn_list_popular.index(isbn)
            distances, indices = index.search(
                popular_embeddings[idx].reshape(1, -1), k_neighbors + 1
            )
            recommended_books = [
                [get_title_isbn(isbn_list_popular[i]), distances[0][j]]
                for j, i in enumerate(indices[0])
                if distances[0][j] > 0 and isbn_list_popular[i] != isbn
            ]
        elif isbn in isbn_list_unpopular:
            if tfidf_embeddings.shape[0] <= 0:
                pp("No valid TF-IDF embeddings. Falling back to most popular books.")
                return get_most_popular(isbn, k_neighbors)
            idx = isbn_list_unpopular.index(isbn)
            distances, indices = index_unpopular.search(
                normalize(tfidf_embeddings[idx].reshape(1, -1)), k_neighbors + 1
            )
            recommended_books = [
                [get_title_isbn(isbn_list_unpopular[i]), distances[0][j]]
                for j, i in enumerate(indices[0])
                if distances[0][j] > 0 and isbn_list_unpopular[i] != isbn
            ]
        else:
            return get_most_popular(isbn, k_neighbors)
    except Exception as e:
        pp("ERROR")
        pp(str(e.with_traceback()))
        return get_most_popular(isbn, k_neighbors)
    return [isbn, recommended_books]


def get_most_popular(isbn, k_neighbors=5):
    # Fallback to most popular books
    recommended_books = [
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
    return [isbn, recommended_books]


def get_title_isbn(isbn):
    return [
        combine_book_ratings[combine_book_ratings["ISBN"] == isbn]["Book-Title"].values[
            0
        ],
        isbn,
    ]


# Test the function
pp(get_recommends("1558745157"))
pp(get_recommends("0330281747"))
