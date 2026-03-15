from pathlib import Path

'''
    File to keep track of easy references to important files in our repo
    Update as needed.
'''

# BotOrNot path file
PROJECT_ROOT = Path(__file__).resolve().parent

# Important Files
KNN = PROJECT_ROOT / 'Model' / 'KNN.py'
MANAGER = PROJECT_ROOT / 'Model' / 'manager.py'
GRAPH = PROJECT_ROOT / 'Model' / 'graph.py'

# Datasets
TWITTER_HUMAN_DATASET = PROJECT_ROOT / 'Datasets' / 'twitter-human-bots-english.csv'

TWIBOT_DATASET = PROJECT_ROOT / 'Datasets' / 'twiBot22_v3.csv'

COMBINED_TWIBOT_DATASET = PROJECT_ROOT / 'Datasets' / 'combined_twiBot_output.csv'