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
user_rating_books_ds = user_rating_books_ds[Feature_List]


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
        self.trainset, self.testset = train_test_split(data,test_size=0.2,random_state=42)
        
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
        return recommendations[:num_recommendations]        
    


# Create Model

recommender = RecommenderPipeline()

# Load data into the pipeline
recommender.load_data(user_rating_books_ds)

# Train the model
recommender.train()

#Evaluate Model
recommender.evaluate()

recommender.trainset.all_users()



users_in_trainset=recommender.trainset.all_users()
user_ex=recommender.trainset.to_raw_uid(24)

# Get recommendations for a user
recommendations = recommender.recommend(user_ex, 50)
print(f"Recommendations for {user_ex}: {recommendations}")

pd.DataFrame(recommendations)
