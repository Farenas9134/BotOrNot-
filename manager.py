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
    # extracted_twitter_dataset_df = setFeaturesdf(twitter_dataset_csv_df)

    extracted_twitter_dataset_df = clean_dataframe

    # removing feature for now. Include maybe later
    extracted_twitter_dataset_df = extracted_twitter_dataset_df.drop(['tweet_repeated_words'], axis = 1)

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
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit(X_scaled)
    
    plt.figure(figsize=(8,6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Original Data (First Two Features)")
    plt.colorbar(label="Diagnosis")
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Transformed Data")
    plt.colorbar(label="Diagnosis")
    plt.show()

    # training / test data based off pca modification
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)



    # WIP KNN class stuff
    tweet_KNN = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 25)}
    knn_gscv = GridSearchCV(tweet_KNN, param_grid, cv=5)
    knn_gscv.fit(X_train, y_train)
    print(knn_gscv.best_params_)
    print("Training accuracy", knn_gscv.best_score_)


    # cv_scores = cross_val_score(tweet_KNN, X_train, y_train, cv=5)

    # tweet_KNN.fit(X_train, y_train)

    

    print("Accuracy is:", knn_gscv.score(X_test, y_test))

    # To be continued. Need to extract features from tweets
