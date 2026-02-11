import pandas as pd

def main():
    # Loads in the dataset
    df = pd.read_csv("hf://datasets/airt-ml/twitter-human-bots/twitter_human_bots_dataset.csv")

    print(df.head())

main()