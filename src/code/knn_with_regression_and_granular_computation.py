# Includes the needed libraries / packages.
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import warnings

# Suppress the warnings.
warnings.filterwarnings('ignore')

# Load the data.
all_ratings = pd.read_csv('src\\data\\Ratings.csv')
all_books = pd.read_csv('src\\data\\Books.csv')
all_users = pd.read_csv('src\\data\\Users.csv')
print(f"Head records of Books:\n{all_books.head()}\nHead records of users:\n{all_users.head()}\nHead records of ratings:\n{all_ratings.head()}\n")

# Merge the ratings data with users on 'User-ID' to get ages alongside ratings.
merged_data = pd.merge(all_ratings.dropna(), all_users[['User-ID', 'Age']], on='User-ID', how='inner')
print(f"Merged data:\n{merged_data}\n")

# Drop NaN values from merged data (both Book-Rating and Age)
# here 40% of readers' data is permenently lost because we do not know the age of these readers.
print(merged_data.info())
merged_data = merged_data.fillna(value=-1)
print(merged_data.info())
print(f"Merged cleaned of NAN values data:\n{merged_data}")

# Dimensions to be checkd for correlation by Pearson method before going 
# to regression recommendation approach.
x_ratings = np.array(merged_data['Book-Rating'])
y_ages = np.array(merged_data['Age'])

# Calculate Pearson correlation coefficient
correlation, p_value = pearsonr(x_ratings, y_ages)

print(f"Pearson Correlation: {correlation}")
print(f"P-value: {p_value}") # equals to zero => no linear relationship between readers' age and books' rating.

# No matter the lack of relationship, we continue with preparation for regression approach
# with experimental aims. So, firstly we shpuld prepare the data sets. 

# Prepare features and target.
X = np.column_stack((merged_data["Book-Rating"].values, merged_data["Age"].values))  # Features: Book Rating and Age
y = merged_data["Book-Rating"].values  # Target: Book Rating
print(f"X column stack of Book Rating and Age recommendation features:\n{X}\n")
print(f"y array of Book Ratings:\n{X}\n")

# Split into train and test datasets.
# The test size is fixed to 0.2 because the datasets are large and we should be sure that
# the model will have enough data to learn patterns efficiently.
# On the other hand the random_state is fixed ensures reproducibility by controlling the 
# randomness of the data splitting. This is important for consistent and easy to debug 
# results during experiments.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train data set:\n{X_train}\n")
print(f"y_train data set:\n{y_train}\n")
print(f"X_test data set:\n{X_test}\n")
print(f"y_test data set:\n{y_test}\n")

# Define the KNN regressor model with the Euclidien metric.
knn = KNeighborsRegressor(n_neighbors=5, metric='euclidean')

# Train the model with the train data sets.
knn.fit(X_train, y_train)

# Granular distance function.
def granular_distance(weights, X1, X2):
    """
    Calculate the weighted granular Euclidean distance between two feature vectors.
    """
    distances = (X1 - X2) ** 2
    weighted_distances = weights * distances
    return np.sqrt(np.sum(weighted_distances))

# Recommend books function.
def recommend_books(user_data, k=5, weights=np.array([2, 1])): # address that the book's rating will be taken with 
                                                               # bigger priority than the reader's age
    """
    Recommend books based on granular similarity using weighted Euclidean distance.
    Returns book title and author.
    """
    # Merge all_ratings and all_users data to get corresponding ratings and ages.
    merged_data = pd.merge(all_ratings, all_users[['User-ID', 'Age']], on='User-ID', how='inner')
    
    # Drop NaN values to avoid issues.
    merged_data = merged_data.dropna()
    
    # Calculate the features for all books (ratings, ages).
    all_books_features = np.column_stack((merged_data["Book-Rating"].values, merged_data["Age"].values))
    
    # Calculate the granular distance for each book using the provided weights.
    distances = np.array([granular_distance(weights, user_data, book_data) for book_data in all_books_features])
    
    # Find the k nearest books based on granular distance.
    indices = np.argsort(distances)[:k]
    
    # Use the indices to get book IDs from merged_data. 
    # The books unique identifiers are their ISBN numbers.
    book_ids = merged_data.iloc[indices]['ISBN'].values  
    
    # Prepare recommendations ad book's title and author.
    recommendations = []
    for book_id in book_ids:
        book_info = all_books[all_books['ISBN'] == book_id].iloc[0]  # Fetch the book details by ISBN
        book_title = book_info['Book-Title']
        book_author = book_info['Book-Author']
        recommendations.append((book_title, book_author))
    
    return recommendations

# Examples of recommending for a new user:
# Example 1: A user who is 33 years old wants to read a book with rating 3.
new_user = np.array([[3, 33]])  
recommendations = recommend_books(new_user)

print("Recommendations for a 33 years old user who wants to read a book with rating 3:")
for rec in recommendations:
    print(f"Book Title: {rec[0]}, Author: {rec[1]}")
print()

# Example 2: A user who is 25 years old wants to read a book with rating 5.
new_user2 = np.array([[5, 25]])  
recommendations2 = recommend_books(new_user2)

print("Recommendations for a 25 years old user who wants to read a book with rating 5:")
for rec in recommendations2:
    print(f"Book Title: {rec[0]}, Author: {rec[1]}")
print()

# Generate predictions for evaluating the model.
y_pred = knn.predict(X_test)

# Evaluation of recommendation quality in percentages.
def precision_at_k(y_true, y_pred, k):
    relevant = set(y_true[:k])  # Top-k true ratings.
    recommended = set(y_pred[:k])  # Top-k predicted ratings.
    return len(relevant & recommended) / k

def recall_at_k(y_true, y_pred, k):
    relevant = set(y_true)  # All true ratings.
    recommended = set(y_pred[:k])  # Top-k predicted ratings.
    return len(relevant & recommended) / len(relevant)

# Example metrics calculation.
y_true = y_test[:10]  # Top-10 true ratings (or actual ratings for evaluation).
y_pred_top_k = y_pred[:10]  # Top-10 predicted ratings.

precision = precision_at_k(y_true, y_pred_top_k, k=5)
recall = recall_at_k(y_true, y_pred_top_k, k=5)

# For example Precision@5 = 0.4 means that, among the top 5 recommendations made 
# by the model, 40% were relevant (i.e., they matched the true ratings or user 
# preferences).
print(f"Precision@5: {precision}")

# For example Recall@5 = 0.5 means that, of all the relevant items for the user, 
# 50% were included in the top 5 recommendations.
print(f"Recall@5: {recall}")

# As we saw earlier this model is not suitable for aour task because there is no linear
# relationship between the considered books' recommendation components. 