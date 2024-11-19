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

data = Dataset.load_from_df(user_rating_books_ds[Feature_List], Reader(rating_scale=(0,10)))

trainset, testset = train_test_split(data, test_size=0.25)

algo = SVD()

algo.fit(trainset)
predictions = algo.test(testset)

accuracy.rmse(predictions)