import pandas as pd

'''
    File to extract features from a column of tweets
'''

def setFeaturesdf(tweet_column):
    rows = []

    for tweet in tweet_column['description']:
        addTweet(tweet, rows)

    df = pd.DataFrame(rows)
    df.to_csv('output', index=False)

    return df

def addTweet(tweet, rows):
    row = extractTweet(tweet)
    rows.append(row)
    return

def extractTweet(tweet):
    tweet_Feature = {'tweet_length':0, "vulgar":0}
    tweet_Feature["tweet_length"] = len(tweet)

    return tweet_Feature