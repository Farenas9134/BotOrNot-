from paths import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Model.extract import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score, matthews_corrcoef
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler

if __name__ == "__main__":

    # Load in dataframe
    clean_dataframe = pd.read_csv("combined_twiBot_output.csv")

    # Extract temporal features before dropping timestamp columns
    clean_dataframe['tweet_created_at'] = pd.to_datetime(clean_dataframe['tweet_created_at'], utc=True)
    clean_dataframe['created_at'] = pd.to_datetime(clean_dataframe['created_at'], utc=True)

    clean_dataframe['tweet_hour_of_day'] = clean_dataframe['tweet_created_at'].dt.hour
    clean_dataframe['tweet_day_of_week'] = clean_dataframe['tweet_created_at'].dt.dayofweek
    clean_dataframe['account_age_at_tweet'] = (
        clean_dataframe['tweet_created_at'] - clean_dataframe['created_at']
    ).dt.days

    # Drop raw timestamps, redundant and unused features
    extracted_twitter_dataset_df = clean_dataframe.drop(
        ['tweet_repeated_words', 'tweet_created_at', 'created_at', 'account_age_days'], axis=1
    )

    # Impute NaNs with median for ratio columns
    for col in ['reputation', 'followers_friends_ratio']:
        extracted_twitter_dataset_df[col] = extracted_twitter_dataset_df[col].fillna(
            extracted_twitter_dataset_df[col].median()
        )

    # Move label to end
    label_column = extracted_twitter_dataset_df.pop('label')
    extracted_twitter_dataset_df.insert(len(extracted_twitter_dataset_df.columns), 'label', label_column)

    print("Human/bot counts:", clean_dataframe['label'].value_counts())
    print("Columns:", extracted_twitter_dataset_df.columns.tolist())

    # Split into features and labels
    X = extracted_twitter_dataset_df.iloc[:, :-1]
    y = extracted_twitter_dataset_df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training data shape:', X_train.shape)
    print('Test data shape:', X_test.shape)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Wrap back into DataFrames to preserve column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

    # Drop low importance features identified by quick RF scan
    features_to_drop = ['protected', "name_contains_'bot'", 'verified',
                        'tweet_question_count', 'tweet_unique_url_count',
                        'tweet_url_count', 'tweet_exclamation_count']

    X_train_reduced = X_train_scaled_df.drop(features_to_drop, axis=1)
    X_test_reduced  = X_test_scaled_df.drop(features_to_drop, axis=1)

    # Undersample majority class (sampling_strategy=0.5 → bots = 50% of humans)
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train_reduced, y_train)

    print("After undersampling:", pd.Series(y_resampled).value_counts().to_dict())

    # Train Random Forest
    final_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    final_rf.fit(X_resampled, y_resampled)

    y_probs = final_rf.predict_proba(X_test_reduced)[:, 1]

    # Evaluating predicted values with different threshold values
    threshold_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    balanced_accuracy_score_list = []
    mcc_score_list = []
    
    for value in threshold_values:
        y_pred = (y_probs >= value).astype(int)
        balanced_accuracy_score_list.append(balanced_accuracy_score(y_test, y_pred))
        mcc_score_list.append(matthews_corrcoef(y_test, y_pred))
    
    # Evaluate
    # y_pred = final_rf.predict(X_test_reduced)
    # print(classification_report(y_test, y_pred, target_names=['bot', 'human']))
    # print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
    # print("MCC:", matthews_corrcoef(y_test, y_pred))

    x = np.arange(len(threshold_values))
    width = 0.35

    fig,ax = plt.subplots()
    rectangles1 = ax.bar(x - width/2, balanced_accuracy_score_list, width,
                         label="Balanced Score", color="blue")
    rectangles2 = ax.bar(x + width/2, mcc_score_list, width,
                         label="MCC Score", color="orange")

    ax.set_title("Accuracy and MCC for Various Thresholds")
    ax.set_xlabel("Threshold Value")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_values)
    ax.legend(loc="upper left")
    plt.show()