import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

user_data = pd.read_csv("data/Users.csv")
rating_data = pd.read_csv("data/Ratings.csv")
book_data = pd.read_csv("data/Books.csv")

user_rating_ds = pd.merge(user_data, rating_data, on="User-ID")
user_rating_books_ds = pd.merge(user_rating_ds, book_data, on="ISBN")

Feature_List = ["User-ID", "Book-Title", "Book-Rating"]
user_rating_books_ds = user_rating_books_ds[Feature_List]

# Encode users and books
user_encoder = LabelEncoder()
book_encoder = LabelEncoder()

user_rating_books_ds.loc[:, "User-ID"] = user_encoder.fit_transform(user_rating_books_ds["User-ID"]).astype(np.int32)
user_rating_books_ds.loc[:, "Book-Title"] = book_encoder.fit_transform(user_rating_books_ds["Book-Title"]).astype(np.int32)
user_rating_books_ds.loc[:, "Book-Rating"] = user_rating_books_ds["Book-Rating"].astype(np.int8)

# Create user-item interaction matrix
interaction_matrix = csr_matrix(
    (user_rating_books_ds["Book-Rating"],
     (user_rating_books_ds["User-ID"], user_rating_books_ds["Book-Title"]))
)

# Train-test split
train_ratio = 0.8
train_size = int(interaction_matrix.shape[0] * train_ratio)
train_data = interaction_matrix[:train_size].astype(np.float32)
test_data = interaction_matrix[train_size:].astype(np.float32)

# Train SVD model using Scikit-learn's TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=42)
svd.fit(train_data)

# Evaluate the model
train_reconstruction = svd.inverse_transform(svd.transform(train_data))
print(train_reconstruction.shape)

#rmse = np.sqrt(mean_squared_error(train_data.toarray(), train_reconstruction))
#print(f"Train RMSE: {rmse}")

# need to build recommend()
# Example: Recommend books for a specific user

#user_id = 10
#num_recommendations = 5
#print(f"Recommendations for User {user_id}: {recommend(user_id, num_recommendations)}")