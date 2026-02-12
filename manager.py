from paths import *
import pandas as pd
import numpy as np
import random
from Model.extract import *
from sklearn.model_selection import train_test_split

'''
    Main file for loading in data, creating model, and training model
'''

if __name__ == "__main__":

    # Load in dataframe
    twitter_dataset_csv_df = pd.read_csv(TWITTER_HUMAN_DATASET)

    # Checking each column feature, cvs is nasty
    # print(twitter_dataset_csv_df.columns)

    # Extract features from each tweet and replace tweet column with these new extracted features
    tweet_column = twitter_dataset_csv_df.loc[:, 'description':'description']
    extracted_tweet_feature_df = setFeaturesdf(tweet_column)


    # Split dataset into features and labels
    X = twitter_dataset_csv_df.loc[:, 'Unnamed: 0':'account_age_days']
    y = twitter_dataset_csv_df.loc[:, 'account_type':'account_type']

    # Split dataset into X_train, y_train, X_test, Y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # See if shapes match expected output
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # To be continued. Need to extract features from tweets
