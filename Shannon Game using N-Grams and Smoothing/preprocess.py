import re
from collections import Counter
import string
import numpy as np


def clean(data):
    """
    Cleans the data by removing the URLs, email addresses, mentions, hashtags, dates, punctuations, etc.
    """
    # print("> Cleaning data...")
    text = data

    text = text.lower()  # convert to lower case

    # regex to find the URLs starting with "http" and replace them with "<URL>"
    text = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)

    # regex to find the URLs starting with "www" and replace them with "<URL>"
    text = re.sub(r'www\.[a-z0-9]+.[a-z]+', '<URL>', text)

    # convert email addresses to <EMAIL>
    text = re.sub(
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '<EMAIL>', text)

    # regex to find all the mentions with @ and replace them with "<MENTION>"
    text = re.sub(r'@[a-zA-Z0-9_]+', '<MENTION>', text)

    # regex to find all the hashtags with # and replace them with "<HASHTAG>"
    text = re.sub(r'#[a-zA-Z0-9_]+', '<HASHTAG>', text)

    # regex to convert "Chapter 1" to "<CHAPTER>"
    text = re.sub(r'Chapter \d+', '<CHAPTER>', text)

    # there are some words which have underscores in the starting and ending, replacing them with just the word
    # remove the underscores from the whole text
    text = re.sub(r'_', '', text)

    # regex to convert all the dates to "<DATE>" of the format dd/mm/yyyy
    text = re.sub(r'\d{1,2}\/\d{1,2}\/\d{2,4}', '<DATE>', text)

    # regex to convert all the dates in the format of "<Date> <Month> <Year>" to "<DATE>"
    text = re.sub(r'\d{1,2} [a-zA-Z]{3,9}  \d{2,4}', '<DATE>', text)

    # regex to convert all the dates in the format of "<Month> <Date>  <Year>" to "<DATE>"
    text = re.sub(r'[a-zA-Z]{3,9} \d{1,2}  \d{2,4}', '<DATE>', text)

    # regex to remove Mr. and Mrs.
    text = re.sub(r'Mr\.|Mrs\.', '', text)

    # replace \n with space
    text = re.sub(r'\n', ' ', text)

    punctuations_to_be_retained = ['#', '@', '<', '>', "'", '"']
    punctuations_to_be_removed = ''.join(
        [p for p in string.punctuation if p not in punctuations_to_be_retained])
    text = re.sub(r'['+punctuations_to_be_removed+']', r' \g<0> ', text)

    text = re.sub(r'\'|\"', '', text)  # remove ' and "
    text = re.sub(r'\s+', ' ', text)  # remove extra empty lines
    # print("> Done cleaning data.")

    return text


def convert_to_sentences(data):
    """
    Converts the data into sentences
    """
    # split the text into sentences
    data = clean(data)
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', data)

    # remove the empty sentences
    sentences = [s.strip() for s in sentences if len(s) > 0]
    # remove the sentences with only space
    sentences = [s for s in sentences if len(s) > 0]
    # remove the sentences with only two spaces
    sentences = [s for s in sentences if s]
    # remove the sentences with only one word
    sentences = [s for s in sentences if len(s.split()) > 1]
    return sentences


def split_dataset(filepath):
    """
    Splits a list (dataset) into
    training and testing data
    """
    with open(filepath, 'r') as f:
        corpus = f.read()

    sentences = convert_to_sentences(corpus)
    test = np.random.choice(sentences, 1000, replace=False).tolist()
    train = [s for s in sentences if s not in test]

    return (train, test)


def create_vocab(data):

    all_words = []
    for sentence in data:
        all_words.extend(sentence.split())

    counted = Counter(all_words)
    return counted


def convert_to_sentence_tokens(corpus, n):
    """
    This function converts the corpus into a list of sentences, where each sentence is a list of tokens
    :param corpus: the corpus
    :return: list of sentences, where each sentence is a list of tokens
    """
    # print("> Converting corpus into a list of sentences, where each sentence is a list of tokens...")
    sentences_tokens = []
    intial_tokens = "<START> "*(n-1)
    intial_tokens = intial_tokens.split()
    sentences_tokens = sentences_tokens + \
        [intial_tokens+s.split()+["<END>"] for s in corpus]

    # print("\tNumber of sentences: ", len(sentences_tokens))
    # print("> Done converting corpus into a list of sentences, where each sentence is a list of tokens.")

    return sentences_tokens
