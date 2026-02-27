from paths import *
import pandas as pd
import numpy as np
import random
from Model.extract import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

'''
    Main file for loading in data, creating model, and training model
'''

if __name__ == "__main__":

    # Load in dataframe
    twitter_dataset_csv_df = pd.read_csv(TWITTER_HUMAN_DATASET)

    # Checking each column feature, cvs is nasty
    print(twitter_dataset_csv_df.columns)

    # Appends columns of extracted tweets and names as a df of shape (N, D + E), where E is the total features extracted
    extracted_twitter_dataset_df = setFeaturesdf(twitter_dataset_csv_df)

    # Split dataset into features and labels
    X = extracted_twitter_dataset_df.loc[:, 'created_at':"name_contains_'bot'"]
    y = extracted_twitter_dataset_df.loc[:, "name_contains_'bot'":'account_type']

    # Split dataset into X_train, y_train, X_test, Y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # See if shapes match expected output
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # WIP KNN class stuff
    # tweet_KNN = KNeighborsClassifier(n_neighbors=3)
    # tweet_KNN.fit(X_train, y_train)

    # To be continued. Need to extract features from tweets
