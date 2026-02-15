import pandas as pd
import re
import math
from collections import Counter

'''
    File to extract features from a column of tweets
'''

def setFeaturedf(tweet_column):
    '''  
        Creates a pandas data frame of extracted features from tweets
    '''
    rows = []

    for tweet in tweet_column['description']:
        addTweet(tweet, rows)

    df = pd.DataFrame(rows)
    df.to_csv('output', index=False)

    return df

def addTweet(tweet, rows):
    '''
        Adds a tweet into a list of tweets for easy access to extract features
    '''
    row = extractTweet(tweet)
    rows.append(row)
    return

def extractTweet(tweet):
    '''
        Extracts the following features: 
    '''
    tweet_Feature = {"tweet_length":0, "tweet_digits_count":0, "tweet_mean_bigram_freq":0,
                     "tweet_entropy":0, "tweet_'bot'_count":0, "tweet_hashtag_count":0,
                     "tweet_url_count":0, "tweet_unique_url_count":0, "tweet_unique_mention_count":0,
                     "tweet_fraction_lowercase_words":0, "tweet_fraction_uppercase_words":0, "tweet_word_count":0,
                     }

    # Extract tweet length
    tweet_Feature["tweet_length"] = len(tweet)

    # Extract Tweet Digits count
    tweet_Feature["tweet_digits_count"] = len(re.findall(r"[0-9]", tweet))
    
    # Extract tweet mean bigram frequency
    bigrams = getWordBigrams(tweet)
    total_bigrams = sum(bigrams.values)
    unique_bigrams = len(bigrams.keys())
    tweet_Feature["tweet_mean_bigram_freq"] = total_bigrams / unique_bigrams
    
    # Extract tweet entropy
    tweet_Feature["tweet_entropy"] = compute_entropy(tweet)

    # Extract tweet "bot" count
    bot_count = re.findall(r"[bB]ot", tweet)
    tweet_Feature["tweet_'bot'_count"] = bot_count

    # Extract tweet "#" count
    hashtag_count = len(re.findall(r"#", tweet))
    tweet_Feature["tweet_hashstag_count"] = hashtag_count

    # Extract tweet url count & unique urls
    urls = re.findall(r"https?://[A-Za-z0-9._~:/?#[\]@!$&'()*+,;=%-]+", tweet)
    urls_dict = Counter(urls)
    tweet_Feature["tweet_url_count"] = len(urls)
    tweet_Feature["tweet_unique_url_count"] = len(urls_dict.keys())

    # Extract unique mentions count
    mentions = Counter(re.findall(r"@[A-Za-z_]+", string))
    tweet_Feature["tweet_unique_mention_count"] = len(mentions.keys())

    # Find instances of all lower, upper, and all words in tweet
    lower_case_words = re.findall(r"\b[a-z]+(?:[-'][a-z]+)?\b", string)
    upper_case_words = re.findall(r"\b[A-Z]+(?:[-'][A-Z]+)?\b", string)
    all_words = lower_case_words + upper_case_words

    # Extract fraction of words in lowercase
    # add edge case in case all words are not 0
    tweet_Feature["tweet_fraction_lowercase_words"] = len(lower_case_words) / len(all_words)

    # Extract fraction of words in uppercase
    tweet_Feature["tweet_fraction_uppercase_words"] = len(upper_case_words) / len(all_words)


    return tweet_Feature

def getCharBigrams(given_string):
    ''' Returns a dictionary of bigrams and their counts for a given string'''
    bigrams = {}

    for index in range(len(given_string) - 1):
        bigram = given_string[index: index + 2]
        if bigram not in bigrams:
            bigrams[bigram] = 1
        else:
            bigrams[bigram] = 0

    return bigrams

def getWordBigrams(given_string):
    ''' Returns a dictionary of bigrams and their counts for a given string'''
    bigrams = {}
    
    words = re.split(r'[: ,]', given_string) # regex is amazing

    for index in range(len(words) - 1):
        bigram1 = words[index]
        bigram2 = words[index + 1]
        bigram = bigram1 + ' ' +  bigram2

        if bigram not in bigrams:
            bigrams[bigram] = 1
        else:
            bigrams[bigram] = 0

    return bigrams

def compute_entropy(string):
    ''' Computes the Shannon entropy for any string (predictability of something)
        low entropy = very predictable | high entropy = low predictability
    '''
    string_length = len(string)

    # get each unique char in string
    chars = {}
    for char in string:
        if char not in chars:
            chars[char] = 1
        else:
            chars[char] += 1

    # compute prob. for each unique char
    for char, value in chars.items():
        chars[char] = value / string_length
    
    # compute entropy!
    entropy = 0
    for char, prob in chars.items():
        entropy += prob * math.log2(prob)
    
    # negate sum (part of Shannon entropy formula)
    entropy = -entropy

    return entropy

if __name__ == "__main__":
    string = "hello well-known I am don't call me bot Bot bot bot HAHAHAHA Yeah #amazing # yeah #YEAHHHAomg http://yeah https://oops @Poop912"
    bigrams = getWordBigrams(string)
    total_bigrams = sum(bigrams.values())
    unique_bigrams = len(bigrams.keys())

    bot_count = len(re.findall(r"[bB]ot", string))
    print("Bot count:", bot_count)

    # Actually grabs all hashtag values, but paper only wants # damn
    hashtag_count = re.findall(r"#[A-Za-z]*", string)
    print("Hashtag count:", hashtag_count)
    
    # okay did search up all possible chars for a url, but I get regex works & why this works
    url_count = re.findall(r"https?://[A-Za-z0-9._~:/?#[\]@!$&'()*+,;=%-]+", string)
    urls = Counter(url_count)
    print("Urls count:",len(urls.items()))

    mentions = Counter(re.findall(r"@[A-Za-z_]+", string))
    print("mentions count:", mentions)

    lower_case_words = re.findall(r"\b[a-z]+(?:[-'][a-z]+)?\b", string)
    print("lower:", lower_case_words)

    upper_case_words = re.findall(r"\b[A-Z]+(?:[-'][A-Z]+)?\b", string)
    print("upper:", upper_case_words)

    all_words = lower_case_words + upper_case_words
    print("all:", all_words)

    print("Entropy:", compute_entropy(string))
