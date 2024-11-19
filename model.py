import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

user_data= pd.read_csv("./data/Users.csv")
rating_data= pd.read_csv("./data/Ratings.csv")
books_data= pd.read_csv("./data/Books.csv")

#print(user_data)
#print(rating_data)
#print(books_data)

user_rating_ds= pd.merge(user_data,rating_data,on='User-ID')
user_rating_books_ds= pd.merge(user_rating_ds,books_data,on='ISBN')

#user_rating_books_ds.info()
#user_rating_books_ds.describe()

ratings_per_user = user_rating_books_ds["User-ID"].value_counts()
ratings_per_user.hist(bins=100)

#plt.title("Distribution of Ratings per User")
#plt.show()

#plt.boxplot(ratings_per_user,vert=False)
#plt.title("Box Plot of Distribution of Ratings per User ")
#plt.show()

lower_bound=ratings_per_user.quantile(.10)
upper_bound=ratings_per_user.quantile(.99)

#print(f"lower bound is:{lower_bound}\n")
#print(f"upper bound is:{upper_bound}")

Feature_List=["User-ID","Book-Title","Book-Rating"]
user_rating_books_ds=user_rating_books_ds[Feature_List]
unique_books=len(user_rating_books_ds["User-ID"].unique())
print(f"There are a total of {unique_books} unique books")

