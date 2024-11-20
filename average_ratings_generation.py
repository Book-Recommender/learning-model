import model
import numpy as np
import pandas as pd

#importing model object
test = model.RecommenderPipeline()
test.load_data(model.user_rating_books_ds)

#Getting rating list and ISBN list
rating_df = model.user_rating_books_ds
book_ID_df = model.books_ID_list

#Modifying ISBN list
book_ID_df = book_ID_df.drop_duplicates(subset=['Book-Title'])
book_ID_df = book_ID_df.sort_values(by="Book-Title")
book_ID_df = book_ID_df.reset_index()

#Getting all unique book titles, and putting them into a dataset
b = rating_df['Book-Title']
b = np.unique(b)
aggregate = pd.DataFrame({'Book-Title': b})
aggregate['Rating'] = 0.0

#This function gets the average review score for each book
def avg(title):
        temp = rating_df.loc[rating_df['Book-Title'] == title]
        return np.mean(temp['Book-Rating'])

#Applying function
aggregate['Rating'] = aggregate['Book-Title'].apply(avg)

#Adding ISBN to score aggregate DF
aggregate['ISBN'] = book_ID_df['ISBN']

#Saving to .csv.gz
aggregate.to_csv("./data/average_ratings.csv.gz", sep = '\t')

#When reading the csv, use index_col=0 as a parameter
pd.read_csv("./data/average_ratings.csv.gz", delimiter = '\t', index_col=0)