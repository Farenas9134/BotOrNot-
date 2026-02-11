import pandas as pd
# from googletrans import Translator
# from langdetect import detect, DetectorFactory, LangDetectException

# translator = Translator()

# def is_english(text):
#     result = translator.detect(text)
#     return result.lang == 'en' and result.confidence > 0.9

# print(is_english("I love programming in Python.")) # True
# print(is_english("Me gusta programar en Python.")) # False

def main():
    # Loads in the dataset
    df = pd.read_csv("hf://datasets/airt-ml/twitter-human-bots/twitter_human_bots_dataset.csv")

    df.to_csv("twitter-human-bots.csv", index=False)

    # print(df.head())

    # removed_rows = []
    # for index, row in df.iterrows():
    #     if not is_english(row['description']):
    #         removed_rows.append(index)
    
    # print(removed_rows)

main()