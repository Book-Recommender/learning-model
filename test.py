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
books_ID_list = user_rating_books_ds[["Book-Title", "ISBN"]]
user_rating_books_ds = user_rating_books_ds[Feature_List]

average_ratings = pd.read_csv("./data/average_ratings.csv.gz", delimiter = '\t', index_col = 0)

class RecommenderPipeline:
    #Encodes data and creates user-item matrix
    def __init__(self):
        self.user_rating = user_rating_books_ds
        self.model = TruncatedSVD(n_components=2)
        self.train_data = None
        self.test_data = None
        # Encode users and books
        user_encoder = LabelEncoder()
        book_encoder = LabelEncoder()
        
        self.user_rating.loc[:, "User-ID"] = user_encoder.fit_transform(self.user_rating["User-ID"]).astype(np.int32)
        self.user_rating.loc[:, "Book-Title"] = book_encoder.fit_transform(self.user_rating["Book-Title"]).astype(np.int32)
        self.user_rating.loc[:, "Book-Rating"] = self.user_rating["Book-Rating"].astype(np.int8)

        # Create Title/Rating interaction matrix
        self.interaction_matrix = csr_matrix(
            (user_rating_books_ds["Book-Rating"],
            (user_rating_books_ds["User-ID"], user_rating_books_ds["Book-Title"]))
        )
        
    #Test/Train splits data and trains model using train data
    def fit(self):
        # Train-test split
        train_ratio = 0.8
        train_size = int(self.interaction_matrix.shape[0] * train_ratio)
        self.train_data = self.interaction_matrix[:train_size].astype(np.float32)
        self.test_data = self.interaction_matrix[train_size:].astype(np.float32)
        
        self.model.fit(self.train_data)

    # This function creates k recommendations for a given user 
    def recommend(self,user_id, num_recommendations = 20):
        
        predicted_ratings = self.model.singular_values_
        user_index = self.interaction_matrix.getrow(user_id)
        user_ratings = predicted_ratings[user_index]
        item_indices = np.argsort(user_ratings)[::-1][:num_recommendations]
        
        recommendations = self.interaction_matrix.columns[item_indices]
    
        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommendations = recommendations[:num_recommendations]   

        #Adds ISBN and returns recommendation dataframe 
        recommendations = pd.DataFrame(recommendations)
        recommendations = pd.concat([recommendations, pd.DataFrame(books_ID_list['ISBN'])], join = 'inner', axis = 1)
        recommendations = recommendations.rename(columns = {0 : "Book-Title", 1: "Rating"})
        return recommendations

    #For users with no previous data, recommend a random list of books by taking a proportion of books within each rating range
    def cold_recommend(self, num_recommendations = 20): 
        fours = average_ratings.loc[average_ratings['Rating'] >= 4.0]   # 5%
        fours = fours.loc[fours['Rating'] < 5.0]
        
        fives = average_ratings.loc[average_ratings['Rating'] >= 5.0]   # 10%
        fives = fives.loc[fives['Rating'] < 6.0]
        
        sixes = average_ratings.loc[average_ratings['Rating'] >= 6.0]   # 10%
        sixes = sixes.loc[sixes['Rating'] < 7.0]
        
        sevens = average_ratings.loc[average_ratings['Rating'] >= 7.0]  # 15%
        sevens = sevens.loc[sevens['Rating'] < 8.0]
        
        eights = average_ratings.loc[average_ratings['Rating'] >= 8.0]  # 25%
        eights = eights.loc[eights['Rating'] < 9.0]
        
        nines = average_ratings.loc[average_ratings['Rating'] >= 9.0]   # 35%
        
        #Take a random sample of each sub-dataset of size num_recommendations
        fours = fours.sample(n = num_recommendations)
        fives = fives.sample(n = num_recommendations)
        sixes = sixes.sample(n = num_recommendations)
        sevens = sevens.sample(n = num_recommendations)
        eights = eights.sample(n = num_recommendations)
        nines = nines.sample(n = num_recommendations)
            
        #Taking the proportions of each sub-dataset. Unseeded to ensure different results each time
        fours = fours.sample(frac=0.05)
        fives = fives.sample(frac=0.10)
        sixes = sixes.sample(frac=0.10)
        sevens = sevens.sample(frac=0.15)
        eights = eights.sample(frac=0.25)
        nines = nines.sample(frac=0.35)
        
        #Merge sub-datasets, shuffle final dataset, and return final result
        final_df = pd.concat([fours, fives, sixes, sevens, eights, nines], axis = 0)
        final_df = final_df.sample(frac = 1)
        final_df = final_df.reset_index(drop=True)
        return final_df
    
recommender = RecommenderPipeline()
recommender.fit()

a = recommender.recommend(0)

print(a['Rating'])

# New is the new code


# This is the new stuff
from sklearn.model_selection import train_test_split

# Flatten sparse matrix to 1d array in order to split data into training and testing sets more comprehensively
non_zero_indices = np.transpose(interaction_matrix.nonzero()) 
ratings = interaction_matrix[non_zero_indices[:, 0], non_zero_indices[:, 1]].A1  

train_indices, test_indices = train_test_split(range(len(ratings)), test_size=0.2, random_state=42)

train_matrix = interaction_matrix.copy()
test_matrix = interaction_matrix.copy()

for index in test_indices:
    train_matrix[non_zero_indices[index][0], non_zero_indices[index][1]] = 0


 # Run SVD Algorithm
svd = TruncatedSVD(n_components=2, random_state=42)  
user_factors = svd.fit_transform(train_matrix)
item_factors = svd.components_

predicted_matrix = np.dot(user_factors, item_factors)


# Collect test ratings and predictions
test_ratings = []
predicted_ratings = []

for index in test_indices:
    user_idx, item_idx = non_zero_indices[index]
    test_ratings.append(interaction_matrix[user_idx, item_idx])
    predicted_ratings.append(predicted_matrix[user_idx, item_idx])

# Compute RMSE
rmse = np.sqrt(mean_squared_error(test_ratings, predicted_ratings))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")


def get_top_n_recommendations(predicted_matrix, user_index, top_n=3):
    
    
    # Get the predicted ratings for the user (user_index)
    user_predictions = predicted_matrix[user_index, :]
    
    # Recursively find and select the indices of the top N items (highest predicted ratings)
    top_n_items = np.argsort(user_predictions)[::-1][:top_n]
    decoded_categories = book_encoder.inverse_transform(top_n_items)
    
    return decoded_categories

# Example: Get the top 3 recommended items for User 1 (user_index=0)
user_index = 10871  # User 1
top_n_recommendations = get_top_n_recommendations(predicted_matrix, user_index, top_n=3)

print(f"Top 3 recommended items for User {user_index}: {top_n_recommendations}")
