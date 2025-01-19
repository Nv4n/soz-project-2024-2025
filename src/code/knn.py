import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# Load the data
all_ratings = pd.read_csv("src/data/Ratings.csv", na_values=["null", "nan", ""])
all_books = pd.read_csv(
    "src/data/Books.csv",
    na_values=["null", "nan", ""],
    usecols=["ISBN", "Book-Title", "Book-Author"],
)
all_users = pd.read_csv("src/data/Users.csv", na_values=["null", "nan", ""])

all_books = all_books.fillna("NaN")
all_ratings = all_ratings.dropna()
all_users = all_users.fillna(-1)

# Merge the ratings data with users on 'User-ID' to get ages alongside ratings
merged_data = pd.merge(
    all_ratings, all_users, on="User-ID", how="inner"
)

# Drop NaN values from merged data (both Book-Rating and Age)
merged_data = merged_data.dropna(subset=["Book-Rating", "Age"])

# Prepare features and target
X = np.column_stack(
    (merged_data["Book-Rating"].values, merged_data["Age"].values)
)  # Features: Book Rating and Age
y = merged_data["Book-Rating"].values  # Target: Book Rating

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the KNN regressor
knn = KNeighborsRegressor(n_neighbors=5, metric="euclidean")

# Train the model
knn.fit(X_train, y_train)


# Granular distance function
def granular_distance(weights, X1, X2):
    """
    Calculate the weighted granular Euclidean distance between two feature vectors.
    """
    distances = (X1 - X2) ** 2
    weighted_distances = weights * distances
    return np.sqrt(np.sum(weighted_distances))


# Modified recommendation logic using granular distance
def recommend_books(user_data, k=5, weights=np.array([1, 1])):
    """
    Recommend books based on granular similarity using weighted Euclidean distance.
    Returns book title and author.
    """
    # Merge all_ratings and all_users data to get corresponding ratings and ages
    merged_data = pd.merge(
        all_ratings, all_users[["User-ID", "Age"]], on="User-ID", how="inner"
    )

    # Drop NaN values to avoid issues
    merged_data = merged_data.dropna(subset=["Book-Rating", "Age"])

    # Calculate the features for all books (ratings, ages)
    all_books_features = np.column_stack(
        (merged_data["Book-Rating"].values, merged_data["Age"].values)
    )

    # Calculate the granular distance for each book using the provided weights
    distances = np.array(
        [
            granular_distance(weights, user_data, book_data)
            for book_data in all_books_features
        ]
    )

    # Find the k nearest books based on granular distance
    indices = np.argsort(distances)[:k]

    # here has out of bound exception
    # Prepare recommendations: book title and author
    recommendations = []
    for idx in indices:
        book_title = all_books.iloc[idx]["Book-Title"]
        book_author = all_books.iloc[idx]["Book-Author"]
        recommendations.append((book_title, book_author))

    return recommendations


# Example of recommending for a new user
new_user = np.array([[3, 33]])  # Example: user with rating 3, age 33
recommendations = recommend_books(new_user)

new_user2 = np.array([[5, 25]])  # Example: user with rating 5, age 25
recommendations2 = recommend_books(new_user2)

print("Recommendations for new_user:")
for rec in recommendations:
    print(f"Book Title: {rec[0]}, Author: {rec[1]}")

print("Recommendations for new_user2:")
for rec in recommendations2:
    print(f"Book Title: {rec[0]}, Author: {rec[1]}")

# Generate predictions (this is what was missing before)
y_pred = knn.predict(X_test)


# Evaluation of recommendation quality
def precision_at_k(y_true, y_pred, k):
    relevant = set(y_true[:k])  # Top-k true ratings
    recommended = set(y_pred[:k])  # Top-k predicted ratings
    return len(relevant & recommended) / k


def recall_at_k(y_true, y_pred, k):
    relevant = set(y_true)  # All true ratings
    recommended = set(y_pred[:k])  # Top-k predicted ratings
    return len(relevant & recommended) / len(relevant)


# Example metrics calculation
y_true = y_test[:10]  # Top-10 true ratings (or actual ratings for evaluation)
y_pred_top_k = y_pred[:10]  # Top-10 predicted ratings

precision = precision_at_k(y_true, y_pred_top_k, k=5)
recall = recall_at_k(y_true, y_pred_top_k, k=5)

print(f"Precision@5: {precision}")
print(f"Recall@5: {recall}")
