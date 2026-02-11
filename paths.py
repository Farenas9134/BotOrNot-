from pathlib import Path

'''
    File to keep track of easy references to important files in our repo
    Update as needed.
'''

# BotOrNot path file
PROJECT_ROOT = Path(__file__).resolve().parent

KNN = PROJECT_ROOT / 'Model' / 'KNN.py'
TWITTER-HUMAN-DATASET = PROJECT_ROOT / 'Datasets' / 'twitter-human-bots.csv'

# if __name__ == "__main__":
#     print("HELLO")
#     print(PROJECT_ROOT)