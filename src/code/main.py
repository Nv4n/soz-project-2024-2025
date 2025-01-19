import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint as pp

df_ratings = pd.read_csv('src/data/Ratings.csv',na_values=['null','nan','','0'])
df_ratings=df_ratings.dropna()
df_books=pd.read_csv('src/data/Books.csv',na_values=['null','nan',''],keep_default_na=False,
                    usecols=['ISBN','Book-Title','Book-Author'])
df_books=df_books.fillna('NaN')
df_users=pd.read_csv('src/data/Users.csv',na_values=['null','nan',''],keep_default_na=False)
df_users= df_users.fillna(-1)

# df_ratings['Book-Rating'].value_counts(sort=True).plot(kind='bar')
# plt.plot(df_ratings['Book-Rating'])
# plt.show()
combine_book_ratings=pd.merge(df_ratings,df_books,on='ISBN')
pp(combine_book_ratings.info())
