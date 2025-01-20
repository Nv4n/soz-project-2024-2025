import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint as pp
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


# The line `df_ratings = pd.read_csv("src/data/Ratings.csv", na_values=["null", "nan", ""])` is
# reading data from a CSV file named "Ratings.csv" into a pandas DataFrame called `df_ratings`. The
# `na_values` parameter is specifying a list of values that should be treated as NaN (missing values)
# when reading the data. In this case, the values "null", "nan", and an empty string "" are being
# considered as missing values and will be replaced with NaN in the DataFrame. This helps in handling
# and identifying missing data during data analysis and processing.
df_ratings = pd.read_csv("src/data/Ratings.csv", na_values=["null", "nan", ""])

# This line of code is reading data from a CSV file named "Books.csv" into a pandas DataFrame called
# `df_books`.
df_books = pd.read_csv(
    "src/data/Books.csv",
    na_values=["null", "nan", ""],
    usecols=["ISBN", "Book-Title", "Book-Author"],
)
# The line `df_users = pd.read_csv("src/data/Users.csv", na_values=["null", "nan", ""])` is reading
# data from a CSV file named "Users.csv" into a pandas DataFrame called `df_users`.
df_users = pd.read_csv("src/data/Users.csv", na_values=["null", "nan", ""])

df_books = df_books.fillna("NaN")
# The line `df_books = df_books.fillna("NaN")` is filling missing values in the DataFrame `df_books`
# with the string "NaN". This means that any cells in the DataFrame that contain NaN (missing) values
# will be replaced with the string "NaN".
df_ratings = df_ratings.dropna()
# The line `df_ratings = df_ratings.dropna()` is removing any rows from the DataFrame `df_ratings`
# that contain missing values (NaN values). This operation helps in cleaning the data by eliminating
# rows with incomplete information, which can be important for data analysis and modeling tasks.
# `df_users = df_users.fillna(-1)` is filling missing values in the DataFrame `df_users` with the
# value `-1`. This means that any cells in the DataFrame that contain NaN (missing) values will be
# replaced with `-1`. This approach is commonly used to handle missing data by replacing it with a
# specific value that can be easily identified and handled in further data processing or analysis.
df_users = df_users.fillna(-1)

# This code snippet is performing the following operations:
# The line `combine_book_ratings = pd.merge(df_ratings, df_books, on="ISBN")` is performing a merge
# operation between two DataFrames `df_ratings` and `df_books` based on the common column "ISBN". This
# operation combines the information from both DataFrames into a single DataFrame
# `combine_book_ratings`.
combine_book_ratings = pd.merge(df_ratings, df_books, on="ISBN")
# The line `combine_book_ratings = combine_book_ratings.drop(["Book-Author"], axis="columns")` is
# dropping the column "Book-Author" from the DataFrame `combine_book_ratings`. This operation removes
# the specified column from the DataFrame, effectively eliminating the "Book-Author" information from
# the combined DataFrame. The `axis="columns"` parameter specifies that the operation should be
# performed along the columns (axis 1) of the DataFrame.
combine_book_ratings = combine_book_ratings.drop(["Book-Author"], axis="columns")


# This code snippet is performing the following operations:
book_rating_count = (
    combine_book_ratings.groupby(by=["Book-Title", "ISBN"])["Book-Rating"]
    .count()
    .reset_index()
    .rename(columns={"Book-Rating": "RatingCount"})
)[["ISBN", "Book-Title", "RatingCount"]]


# This line of code is performing a merge operation between two DataFrames `combine_book_ratings` and
# `book_rating_count` based on the common columns "ISBN" and "Book-Title". The `how="left"` parameter
# specifies a left join operation, which means that all the rows from the `combine_book_ratings`
# DataFrame will be retained in the result, and only the matching rows from the `book_rating_count`
# DataFrame will be added based on the specified columns.
book_rating_with_total_count = combine_book_ratings.merge(
    book_rating_count, on=["ISBN", "Book-Title"], how="left"
)

# The code `pp(book_rating_with_total_count["RatingCount"].quantile(np.arange(0.9, 1, 0.01)))` is
# calculating the quantiles of the "RatingCount" column in the DataFrame
# `book_rating_with_total_count`.
pp(book_rating_with_total_count["RatingCount"].quantile(np.arange(0.9, 1, 0.01)))
popularity_threshold = 136

# This line of code `rating_popular_books = book_rating_with_total_count.query("RatingCount >=
# @popularity_threshold")` is filtering the DataFrame `book_rating_with_total_count` to include only
# those rows where the value in the "RatingCount" column is greater than or equal to the
# `popularity_threshold` value.
rating_popular_books = book_rating_with_total_count.query(
    "RatingCount >= @popularity_threshold"
)

# This code snippet is performing the following operations:
pivot = (
    rating_popular_books.drop_duplicates(["Book-Title", "User-ID"])
    .pivot(index="Book-Title", columns="User-ID", values="Book-Rating")
    .fillna(0)
)
# matrix_popular_books = csr_matrix(pivot.values)

# The line `model_knn = NearestNeighbors(metric="cosine", algorithm="auto")` is creating an instance
# of the NearestNeighbors class from the scikit-learn library for performing nearest neighbor search.
# Here's what each parameter in the instantiation means:
model_knn = NearestNeighbors(metric="cosine", algorithm="auto")

# The line `model_knn.fit(csr_matrix(pivot.values))` is fitting the NearestNeighbors model `model_knn`
# with the data represented as a Compressed Sparse Row matrix (CSR matrix). Here's a breakdown of what
# this line of code is doing:
model_knn.fit(csr_matrix(pivot.values))


# TODO LOOK AT ANN with ANNOY OR FAISS
# TODO Look
def get_recommends(book=""):
    """
    The function `get_recommends` takes a book as input, finds the nearest neighbors using a KNN model,
    and returns a list of recommended books along with their distances from the input book.

    @param: book str

    The `get_recommends` function takes a book title as input and returns a list of
    recommended books based on a KNN model. The function first locates the book in a pivot table, then
    uses a KNN model to find the nearest neighbors of the book. It then returns a list

    @return: recomended_books

    The `get_recommends` function returns a list containing the input `book` and a list of
    recommended books along with their distances from the input book. If an error occurs (e.g., the
    input book is not found in the dataset), it returns a message indicating that the book is not in the
    top books.
    """
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


# The code `pp(get_recommends("Tess of the D'Urbervilles (Wordsworth Classics)"))` is calling the
# `get_recommends` function with the input book title "Tess of the D'Urbervilles (Wordsworth
# Classics)" and then pretty-printing the output of the function call.
pp(get_recommends("Tess of the D'Urbervilles (Wordsworth Classics)"))
