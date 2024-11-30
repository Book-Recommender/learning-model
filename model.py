import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

user_data = pd.read_csv("data/Users.csv")
rating_data = pd.read_csv("data/Ratings.csv")
books_data = pd.read_csv("data/Books.csv")

user_rating_ds = pd.merge(user_data,rating_data,on='User-ID')
user_rating_books_ds = pd.merge(user_rating_ds,books_data,on='ISBN')

Feature_List=["User-ID","Book-Title","Book-Rating"]
books_ID_list = user_rating_books_ds[["Book-Title", "ISBN"]]
user_rating_books_ds = user_rating_books_ds[Feature_List]

average_ratings = pd.read_csv("./data/average_ratings.csv.gz", delimiter = '\t', index_col = 0)

#Encapsulate Model in classs called "RecommenderPipeline"

class RecommenderPipeline:
    def __init__(self,rating_scale=(0,10)):
        
        #define instance attributes; Initialize model, set rating_scale(needed for reader object) and declare 
        # trainset and test set variables
        self.model=SVD()
        self.rating_scale = rating_scale
        self.trainset= None
        self.testset = None
        
        # This function converts the original pandas dataframe object into surprise format which is neeede
    def load_data(self,df):
        reader= Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(df[Feature_List],reader)
        self.trainset, self.testset = train_test_split(data,test_size=0.2)
        
        # train model, and raise error if dataset is not provided yet
    def train(self):
        if not self.trainset:
            raise ValueError("No training data found. Please load data first with load_data function.")
        self.model.fit(self.trainset)
        
        # evaluate the model using RMSE, or raise error if dataset hasnt been provided yet
    def evaluate(self):
        if not self.testset:
            raise ValueError("No test data found. Please load data first with load_data function.")
        predictions = self.model.test(self.testset)
        rmse = accuracy.rmse(predictions)
        
        return rmse
       
        # This function creates k recommendations for a given user 
    def recommend(self,user_id,num_recommendations):
        user_ratings = self.trainset.ur[self.trainset.to_inner_uid(user_id)]
        all_items = set(self.trainset.all_items())
        rated_items = set(item for item, _ in user_ratings)
        unrated_items = all_items - rated_items
        
        recommendations = [(self.trainset.to_raw_iid(item), self.model.predict(user_id, self.trainset.to_raw_iid(item)).est)
            for item in unrated_items]
    
        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommendations = recommendations[:num_recommendations]   

        #Adds ISBN and returns recommendation dataframe 
        recommendations = pd.DataFrame(recommendations)
        recommendations = pd.concat([recommendations, pd.DataFrame(books_ID_list['ISBN'])], join = 'inner', axis = 1)
        recommendations = recommendations.rename(columns = {0 : "Book-Title", 1: "Rating"})
        return recommendations
    
        # For users with no previous data, recommend a random list of books by taking a proportion of books within each rating range
    def cold_recommend(self, num_recommendations): 
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

def book_recommendations(num_recommendations):
    
    # Create Model
    recommender = RecommenderPipeline()
    
    #If there is no user data, then use cold_recommend()
    #For now, since data cannot be pulled directly from database, a simple boolean is used here
    is_data = True
    
    if not is_data:
        return recommender.cold_recommend(num_recommendations)
    
    # Load book data into the pipeline if there is user data
    recommender.load_data(user_rating_books_ds)

    # Train the model
    recommender.train()

    user_ex=recommender.trainset.to_raw_uid(24)

    # Get recommendations for a user
    return recommender.recommend(user_ex, num_recommendations)




# # Create Model

# recommender = RecommenderPipeline()

# # Load data into the pipeline
# recommender.load_data(user_rating_books_ds)

# # Train the model
# recommender.train()

# #Evaluate Model
# recommender.evaluate()

# recommender.trainset.all_users()

# users_in_trainset=recommender.trainset.all_users()
# user_ex=recommender.trainset.to_raw_uid(24)

# # Get recommendations for a user
# recommendations = recommender.recommend(user_ex, 50)
# print(f"Recommendations for {user_ex}: {recommendations}")