from paths import *
import pandas as pd
import numpy as np
from Model.extract import *

# graph stuff
import matplotlib.pyplot as plt
import seaborn as sns

# scikitlearn ML models
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

'''
    Main file for loading in data, creating model, and training model
'''


if __name__ == "__main__":

    # Load in dataframe
    clean_dataframe = pd.read_csv("combined_output.csv")

    # twitter_dataset_csv_df = pd.read_csv(TWITTER_HUMAN_DATASET)

    # Checking each column feature, cvs is nasty
    # print(twitter_dataset_csv_df.columns)
    print(clean_dataframe.columns)


    # test only fist k rows
    # twitter_dataset_csv_df = twitter_dataset_csv_df.iloc[:10]

    # Appends columns of extracted tweets and names as a df of shape (N, D + E), where E is the total features extracted
    extracted_twitter_dataset_df = clean_dataframe

    # extracted_twitter_dataset_df = setFeaturesdf(extracted_twitter_dataset_df)

    # extracted_twitter_dataset_df = extracted_twitter_dataset_df[:10]

    # removing feature for now. Include maybe later
    extracted_twitter_dataset_df = extracted_twitter_dataset_df.drop(['tweet_repeated_words'], axis = 1)

    counts_human_bot = extracted_twitter_dataset_df['account_type'].value_counts()

    print("Human/bot counts:", counts_human_bot)

    # Split dataset into features and labels
    X = extracted_twitter_dataset_df.loc[:, 'created_at':"name_contains_'bot'"]
    y = extracted_twitter_dataset_df.loc[:, 'account_type']

    # Split dataset into X_train, y_train, X_test, Y_test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # See if shapes match expected output
    # print('Training data shape: ', X_train.shape)
    # print('Training labels shape: ', y_train.shape)
    # print('Test data shape: ', X_test.shape)
    # print('Test labels shape: ', y_test.shape)


    # WIP PCA (reduce dimension of 40 total features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test)


    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.fit_transform(X_test_scaled)

    # training / test data based off pca modification

    # WIP KNN class stuff
    tweet_KNN = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 25)}
    knn_gscv = GridSearchCV(tweet_KNN, param_grid, cv=5)
    knn_gscv.fit(X_train_pca, y_train)
    print(knn_gscv.best_params_)
    print("Training accuracy", knn_gscv.best_score_)
   
    print("Accuracy is:", knn_gscv.score(X_test_pca, y_test))

    y_pred = knn_gscv.predict(X_test_pca)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Precision is:", precision)
    print("Recall is:", recall)
    
    # To be continued. Need to extract features from tweets
