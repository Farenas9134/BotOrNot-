import pandas as pd
import re

'''
    Counts the number of words in a tweet
'''
def countWords(tweet):
    words = re.findall(r"\b[\w']+\b", tweet)

    length = len(words)
    return length

'''
    Finds the number of hashtags in a tweet
    i.e. #cool but not #
'''
def countHashtag(tweet):
    num_hashtags = 0

    split_tweet = tweet.split()

    for word in split_tweet:
        if word[0] == "#" and len(word) > 1:
            num_hashtags += 1

    return num_hashtags

'''
    Finds the number of mentions in a tweet
    i.e. @coolguy123 but not @
'''
def countMentions(tweet):
    num_mentions = 0

    split_tweet = tweet.split()

    for word in split_tweet:
        if word[0] == "@" and len(word) > 1:
            num_mentions += 1

    return num_mentions

'''
    Finds the number of repeated words in a tweet
'''
# THIS DOESN'T DIFFERENTIATE BETWEEN CAPITAL VS LOWERCASE AS IT IS
def countRepeats(tweet):
    word_dict = {}
    repeated_words = {}

    split_tweet = tweet.split()

    for word in split_tweet:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            if word not in repeated_words:
                repeated_words[word] = 1
            else:
                repeated_words[word] += 1
    
    return repeated_words

def main():
    # df = pd.read_csv("Datasets/twitter-human-bots-english.csv")
    sample_tweet = "Loan coach at @mancity & Aspiring DJ loan loan #soccer #music @"

    # print(countWords(sample_tweet))
    # print(countHashtag(sample_tweet))
    # print(countMentions(sample_tweet))
    print(countRepeats(sample_tweet))



main()