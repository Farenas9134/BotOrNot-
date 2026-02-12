import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0 # for consistent results

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def main():
    # Loads in the dataset
    df = pd.read_csv("Datasets/twitter-human-bots.csv")
    
    # Keeps track of all the entries that have a non-English tweet. 
    removed_rows = []
    for index, row in df.iterrows():
        if not is_english(row['description']):
            removed_rows.append(index)
    

    english_df = df.drop(index=removed_rows)

    english_df.to_csv("twitter-human-bots-english.csv", index=False)


main()