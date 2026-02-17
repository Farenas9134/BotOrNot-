import pandas as pd
import re
import math
from collections import Counter

'''
    File to extract features from a column of tweets
'''

def setFeaturesdf(tweet_column, name_column):
    '''  
        Creates a pandas data frame of extracted features from tweets & names
    '''
    tweet_rows = []
    name_rows = []

    for tweet in tweet_column['description']:
        addTweet(tweet, tweet_rows)
    
    for name in name_column['screen_name']:
        addName(name, name_rows)

    tweet_df = pd.DataFrame(tweet_rows)
    tweet_df.to_csv('tweet_output.csv', index=False)

    name_df = pd.DataFrame(name_rows)
    name_df.to_csv('name_output.csv', index=False)

    return tweet_df, name_df

def addTweet(tweet, rows):
    '''
        Adds a tweet into a list of tweets for easy access to extract features
    '''
    row = extractTweetFeatures(tweet)
    rows.append(row)
    return

def addName(name, rows):
    '''
        Adds a name into a list of names for easy access to extract features
    '''
    row = extractNameFeatures(name)
    rows.append(row)
    return

def extractTweetFeatures(tweet):
    '''
        Extracts the following features: 
    '''
    tweet_Feature = {"tweet_length":0, "tweet_digits_count":0, "tweet_mean_bigram_freq":0,
                     "tweet_entropy":0, "tweet_'bot'_count":0, "tweet_hashtag_count":0,
                     "tweet_url_count":0, "tweet_unique_url_count":0, "tweet_unique_mention_count":0,
                     "tweet_fraction_lowercase_words":0, "tweet_fraction_uppercase_words":0, "tweet_word_count":0,
                     "tweet_sentence_count":0, "tweet_avg_word_length":0, "tweet_avg_word_per_sentence": 0}

    # Extract tweet length
    tweet_Feature["tweet_length"] = len(tweet)

    # Extract Tweet Digits count
    tweet_Feature["tweet_digits_count"] = len(re.findall(r"[0-9]", tweet))
    
    # Extract tweet mean bigram frequency
    bigrams = getWordBigrams(tweet)
    total_bigrams = sum(bigrams.values())
    unique_bigrams = len(bigrams.keys())
    if total_bigrams == 0:
        tweet_Feature["tweet_mean_bigram_freq"] = 0
    else:
        tweet_Feature["tweet_mean_bigram_freq"] = total_bigrams / unique_bigrams
    
    # Extract tweet entropy
    tweet_Feature["tweet_entropy"] = compute_entropy(tweet)

    # Extract tweet "bot" count
    bot_count = re.findall(r"[bB]ot", tweet)
    tweet_Feature["tweet_'bot'_count"] = len(bot_count)

    # Extract tweet "#" count
    hashtag_count = len(re.findall(r"#", tweet))
    tweet_Feature["tweet_hashtag_count"] = hashtag_count

    # Extract tweet url count & unique urls
    urls = re.findall(r"https?://[A-Za-z0-9._~:/?#[\]@!$&'()*+,;=%-]+", tweet)
    urls_dict = Counter(urls)
    tweet_Feature["tweet_url_count"] = len(urls)
    tweet_Feature["tweet_unique_url_count"] = len(urls_dict.keys())

    # Extract unique mentions count
    mentions = Counter(re.findall(r"@[A-Za-z_]+", tweet))
    tweet_Feature["tweet_unique_mention_count"] = len(mentions.keys())

    # Find instances of all lower, upper, and all words in tweet
    lower_case_words = re.findall(r"\b[a-z]+(?:[-'][a-z]+)?\b", tweet)
    upper_case_words = re.findall(r"\b[A-Z]+(?:[-'][A-Z]+)?\b", tweet)
    all_words = lower_case_words + upper_case_words

    # Extract fraction of words in lowercase
    # add edge case in case all words is 0
    try:
        tweet_Feature["tweet_fraction_lowercase_words"] = len(lower_case_words) / len(all_words)
    except:
        tweet_Feature["tweet_fraction_lowercase_words"] = 0

    # Extract fraction of words in uppercase
    try:
        tweet_Feature["tweet_fraction_uppercase_words"] = len(upper_case_words) / len(all_words)
    except:
        tweet_Feature["tweet_fraction_uppercase_words"] = 0

    # Extract count of words in tweet
    tweet_Feature["tweet_word_count"] = len(all_words)

    # Extract count of sentences in tweet
    sentence_count = re.findall(r"[^.!?]+[.!?]+" , tweet)
    tweet_Feature["tweet_sentence_count"] = len(sentence_count)

    # Extract average word length
    unique_words = Counter(all_words)
    total_word_len = 0
    for word in unique_words.keys():
        total_word_len += len(word)
    
    try:
        avg_word_length = total_word_len / len(unique_words.keys())
        tweet_Feature["tweet_avg_word_length"] = avg_word_length
    except:
        tweet_Feature["tweet_avg_word_length"] = 0

    # Extract avgerage words per sentence
    try:
        avg_word_per_sentence = len(all_words) / len(sentence_count)
        tweet_Feature["tweet_avg_word_per_sentence"] = avg_word_per_sentence
    except:
        tweet_Feature["tweet_avg_word_per_sentence"] = 0
    
    return tweet_Feature

def extractNameFeatures(name):
    '''
        Extracts the following features: 
    '''
    name_Features = {"name_length":0, "name_digits_count":0, "name_mean_bigram_freq":0,
                     "name_entropy":0, "name_contains_'bot'":0}
    
    # Extract name length of user profile username
    name_Features["name_length"] = len(name)

    # Extract count of digits in username
    name_Features["name_digits_count"] = len(re.findall(r"[0-9]", name))

    # Extract username mean bigram frequency
    bigrams = getCharBigrams(name)
    total_bigrams = sum(bigrams.values())
    unique_bigrams = len(bigrams.keys())
    if total_bigrams == 0:
        name_Features["name_mean_bigram_freq"] = 0
    else:
        name_Features["name_mean_bigram_freq"] = total_bigrams / unique_bigrams

    # Extract username entropy value
    name_Features["name_entropy"] = compute_entropy(name)

    # Determine if username contains 'bot'
    name_Features["name_contains_'bot'"] = 1 if 'bot' in name.lower() else 0 # lmao cool syntax

    return name_Features

def getCharBigrams(given_string):
    ''' Returns a dictionary of bigrams and their counts for a given string'''
    bigrams = {}

    for index in range(len(given_string) - 1):
        bigram = given_string[index: index + 2]
        if bigram not in bigrams:
            bigrams[bigram] = 1
        else:
            bigrams[bigram] += 1

    return bigrams

def getWordBigrams(given_string):
    ''' Returns a dictionary of bigrams and their counts for a given string'''
    bigrams = {}
    
    words = re.findall(r'\b\w+\b', given_string) # regex is amazing

    for index in range(len(words) - 1):
        bigram1 = words[index]
        bigram2 = words[index + 1]
        bigram = bigram1 + ' ' +  bigram2

        if bigram not in bigrams:
            bigrams[bigram] = 1
        else:
            bigrams[bigram] += 1

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
    string = "hello well-known I am don't call me bot Bot bot bot HAHAHAHA Yeah #amazing # yeah #YEAHHHAomg http://yeah https://oops @Poop912. Yay! omg cool."
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

    sentence_count = re.findall(r"[^.!?]+[.!?]+" , string)
    print("sentences:", sentence_count)
    print("Entropy:", compute_entropy(string))

    unique_words = Counter(all_words)
    total_word_len = 0
    for word in unique_words.keys():
        total_word_len += len(word)
    
    avg_word_len = total_word_len / len(unique_words.keys())
    print("avg word len:", avg_word_len)

    avg_word_per_sentence = len(all_words) / len(sentence_count)
    print("avg words per sentence:", avg_word_per_sentence)
