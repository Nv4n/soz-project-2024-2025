import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint as pp

df_ratings = pd.read_csv("src/data/Ratings.csv", na_values=["null", "nan", "", "0"])
df_ratings = df_ratings.dropna()
df_books = pd.read_csv(
    "src/data/Books.csv",
    na_values=["null", "nan", ""],
    keep_default_na=False,
    usecols=["ISBN", "Book-Title", "Book-Author"],
)
df_books = df_books.fillna("NaN")
df_users = pd.read_csv(
    "src/data/Users.csv", na_values=["null", "nan", ""], keep_default_na=False
)
df_users = df_users.fillna(-1)

# df_ratings['Book-Rating'].value_counts(sort=True).plot(kind='bar')
# plt.plot(df_ratings['Book-Rating'])
# plt.show()
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
# pp(
#     book_rating_with_total_count.sort_values(by="Rating-Count", ascending=False).head(
#         20
#     )
# )

pp(book_rating_with_total_count["RatingCount"].describe())
# pp(book_rating_with_total_count["Book-Rating"].describe())
pp(book_rating_with_total_count["RatingCount"].quantile(np.arange(0.9, 1, 0.01)))

popularity_threshold = 278
rating_popular_books = book_rating_with_total_count.query(
    "RatingCount >= @popularity_threshold"
)

pp(rating_popular_books.info())
