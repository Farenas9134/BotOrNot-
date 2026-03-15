import pandas as pd
import re
import spacy 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

'''
    Helper function that tokenizes tweets by white space
'''

nlp = spacy.load("en_core_web_sm")
sid_obj = SentimentIntensityAnalyzer()

def splitTweet(tweet):

    split_tweet = tweet.split()

    return split_tweet

'''
    Counts the number of words in a tweet
'''
def countWords(tweet):
    words = re.findall(r"\b[\w']+\b", tweet)

    length = len(words)
    return length

'''
    Finds the number of mentions in a tweet
    i.e. @coolguy123 but not @
'''
def countMentions(tweet):
    num_mentions = 0

    split_tweet = splitTweet(tweet)

    for word in split_tweet:
        if word[0] == "@" and len(word) > 1:
            num_mentions += 1

    return num_mentions

'''
    Finds the number of repeated words in a tweet
'''
def countRepeats(tweet):
    word_dict = {}
    repeated_words = {}

    split_tweet = splitTweet(tweet)

    for word in split_tweet:
        if word.lower() not in word_dict:
            word_dict[word.lower()] = 1
        else:
            if word.lower() not in repeated_words:
                # Means that it has shown up twice so far
                repeated_words[word.lower()] = 2
            else:
                repeated_words[word.lower()] += 1
    
    return repeated_words

'''
    Counts the number of questions in a tweet
'''
def questionCount(sentences):
    question_count = 0

    for sentence in sentences:
        if sentence.strip().endswith("?"):
            question_count += 1

    return question_count  

'''
    Counts the number of exclamations in a tweet
'''
def exclamationCount(sentences):
    exclamation_count = 0

    for sentence in sentences:
        if sentence.strip().endswith("!"):
            exclamation_count += 1

    return exclamation_count  

'''
    Returns the number of questions and exclamations in a tweet
'''
def getNumQuestionsAndExclamations(tweet):
    # Splits a tweet into sentences if there is punctuation like ?, !, or .
    sentences = re.split(r'(?<=[?!.])\s+', tweet)

    question_count = questionCount(sentences)
    exclamation_count = exclamationCount(sentences)

    return question_count, exclamation_count

''''
    Counts the number of characters that are not alphanumeric or white space
'''
def countSpecialChars(tweet):
    special_chars = 0

    for char in tweet:
        if not char.isalnum() and not char.isspace():
            special_chars += 1

    return special_chars

'''
    Returns a dictionary with the number of times certain parts of speech show up
    i.e. number of nouns, number of verbs, number of adverbs
'''
def countPartsOfSpeech(tweet):
    doc = nlp(tweet)

    noun_list = []
    adjective_list = []
    verb_list = []
    pronoun_list = []
    adverb_list = []
    for token in doc:
        if token.pos_ == "NOUN":
            noun_list.append(token)
        elif token.pos_ == "ADJ":
            adjective_list.append(token)
        elif token.pos_ == "VERB":
            verb_list.append(token)
        elif token.pos_ == "PRON":
            pronoun_list.append(token)
        elif token.pos_ == "ADV":
            adverb_list.append(token)
        
    parts_of_speech_dict = {"NOUN": len(noun_list), "ADJECTIVE": len(adjective_list), "ADVERB": len(adverb_list), "VERB": len(verb_list), "PRONOUN": len(pronoun_list)}

    return parts_of_speech_dict

'''
    Returns sentiment class of a tweet based off set thresholds
'''
def getSentimentScore(tweet):
    sentiment_dict = sid_obj.polarity_scores(tweet)

    if sentiment_dict['compound'] >= 0.05:
        return "Positive"
    elif sentiment_dict['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"