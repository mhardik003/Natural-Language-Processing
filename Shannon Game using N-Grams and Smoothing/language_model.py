import sys
import random
import re
import math
from collections import Counter
from preprocess import *
import string
import numpy as np

# dictionary with key having the N of N gram
all_n_grams = {1: [], 2: [], 3: [], 4: [], 5: []}
# the number of unique words before which the sequence occurs (the numerator in (1- lambda) expression)
unique_n_grams = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
# dictionary with key having the N of N gram,  which itself is a dictionary of the count of that N-gram (the other denominator in the (1-lambda) expression)
n_gram_counts = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
probabilities = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
all_tokens_vocab = []
all_tokens_in_corpus = []


witten_bell_probabilities = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
log_probabilities_witten = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}

discount = 0.9
kneserney_probabilities = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
log_probabilities_kneserney = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}

def find_probablity(n):
    """
    This function finds the probability of each n-gram in the corpus
    """

    if (n == 1):
        # for unigrams the probability will be the count of the word divided by the total number of words in the corpus
        for n_gram in n_gram_counts[n]:
            probability = n_gram_counts[n][n_gram]/len(all_tokens_in_corpus)
            probabilities[n].update({n_gram: probability})
            # print("Added probability for n_gram: ", n_gram, " with probability: ", probability)
    else:
        # for all other n-gram the probability will be the count of the n-gram divided by the count of the n-1 gram
        for n_gram in n_gram_counts[n]:
            history = n_gram[:(n-1)]
            probability = n_gram_counts[n][n_gram]/n_gram_counts[n-1][history]
            probabilities[n].update({n_gram: probability})


def find_ngrams(n, input_list):  # go inside the list of sentences
    """
    This function finds all the n-grams in the corpus
    """
    for i in range(len(input_list)):  # for each sentence find all the n-grams
        for j in range(len(input_list[i])-n+1):
            ngram = tuple(input_list[i][j:j+n])
            all_n_grams[n].append(ngram)


def common_data(n, train_data, train_data_vocab):

    # print("> Creating common data...")
    sentences_tokens = convert_to_sentence_tokens(train_data, n)

    all_tokens_vocab.extend(train_data_vocab)
    all_tokens_in_corpus.extend(
        [token for s in sentences_tokens for token in s])

    for i in range(1, 5):
        find_ngrams(i, sentences_tokens)
        n_gram_counts[i] = dict(Counter(all_n_grams[i]))
        unique_n_grams[i-1] = dict(Counter([i[:-1]
                                   for i in n_gram_counts[i].keys()]))
        find_probablity(i)

    # print("\tCreated n-grams for n = 1 to n = ", n)
    # print("\tCreated n-gram counts for n = 1 to n = ", n)
    # print("\tCreated unique n-grams for n = 1 to n = ", n)
    # print("\tCreated probabilities for n = 1 to n = ", n)
    # print("> Done creating common data.")


def find_lambda(n, history):
    unique_n_grams[n-1].setdefault(history, 0)
    n_gram_counts[n-1].setdefault(history, 0)
    # if the history is not present in the corpus then we will assign zero probability to the current n-gram and shift all the mass to the next (n-1)-gram
    if ((unique_n_grams[n-1][history]+n_gram_counts[n-1][history]) == 0):
        return 0

    lamda = 1-(unique_n_grams[n-1][history]/(n_gram_counts[n-1]
               [history]+unique_n_grams[n-1][history]))
    return lamda


def witten_bell_for_one(n,  tuple_n):
    history = tuple_n[:(n-1)]
    if (n == 1):
        n_gram_counts[n].setdefault(tuple_n, 1)
        witten_probability = n_gram_counts[n][tuple_n] / \
            len(unique_n_grams[n].keys())
        # witten_probability = n_gram_counts[n][tuple_n]/len(all_tokens_in_corpus) + (1 / (len(all_tokens_in_corpus)+len(all_tokens_vocab)))
        witten_bell_probabilities[n].update({tuple_n: witten_probability})
        return witten_probability

    lamda = find_lambda(n, history)
    probabilities[n].setdefault(tuple_n, (1/len(probabilities[n].keys())))
    witten_probability = probabilities[n][tuple_n] * \
        lamda + (1-lamda)*witten_bell_for_one(n-1, tuple_n[1:])
    witten_bell_probabilities[n].update({tuple_n: witten_probability})

    return witten_probability


def wittenbell(n):
    """
    This function calculates the Witten-Bell probabilities
    """
    for tuples in n_gram_counts[n].keys():
        witten_bell_for_one(n, tuples)


def calculate_log_probability_witten(n):
    for i in range(1, 5):
        for key in witten_bell_probabilities[i].keys():
            log_probabilities_witten[i].update(
                {key: math.log(witten_bell_probabilities[i][key])})


def find_perplexity(n, sentences_tokens, log_probabilities, check=0):
    all_perplexity = []
    if (len(sentences_tokens) != 0):
        for sentence in sentences_tokens:
            single_perplex = 0
            for i in range(len(sentence) - n + 1):
                if tuple(sentence[i:i+n]) in log_probabilities[4].keys():
                    single_perplex += log_probabilities[4][tuple(
                        sentence[i:i+n])]
                else:
                    log_probabilities[4].setdefault(
                        tuple(sentence[i:i+n]), (1/len(log_probabilities[4].keys())))
                    log_probabilities[4][tuple(
                        sentence[i:i+n])] = math.log(witten_bell_for_one(n, tuple(sentence[i:i+n])))
                    single_perplex += log_probabilities[4][tuple(
                        sentence[i:i+n])]
            if(check == 1):
                print("Probability : ",math.exp(single_perplex)) 
            single_perplex /= (len(sentence) - n + 1)
            single_perplex *= -1
            single_perplex = math.exp(single_perplex)
            all_perplexity.append(single_perplex)
    return all_perplexity


def wrapper_wittenBell(n, corpus, input_sentence):
    """
    Creates a Witten-Bell language model
    """

    # print("> Creating Witten-Bell language model...")
    trainset, testset = split_dataset(corpus)

    vocab = create_vocab(trainset)
    # print(vocab)

    common_data(n, trainset, vocab)
    # print(all_n_grams[4][:10])
    # print(n_gram_counts[3][('<START>', '<START>', 'pride')])
    # print(unique_n_grams[2][('it', 'is')])
    # print(probabilities[3][('<START>', '<START>', 'pride')])
    # print("> Initializing Witten-Bell language model...")
    wittenbell(4)
    # print("> Done creating Witten-Bell language model.")
    # print(witten_bell_probabilities[4][('<START>', '<START>', 'pride', '&')])
    # print("> Calculating log probabilities for Witten-Bell language model...")
    calculate_log_probability_witten(n)
    # print("> Done calculating log probabilities for Witten-Bell language model.")
    # print(log_probabilities_witten[4][('is', 'a', 'truth', 'universally')])
    # print("> Calculating perplexity for Witten-Bell language model...")
    testset = convert_to_sentence_tokens(testset, n)
    perplexity = find_perplexity(n, testset, log_probabilities_witten)
    # print("> Done calculating perplexity for Witten-Bell language model.")

    # print("> Calculating average perplexity for Witten-Bell language model...")
    avg_perplexity = sum(perplexity)/len(perplexity)
    # print("> Done calculating average perplexity for Witten-Bell language model.")
    # print("Average perplexity for Witten-Bell language model is: ", avg_perplexity)
    print("Average perplexity : ",avg_perplexity)
        
    if ('Pride' in corpus):
      with open ('./Results/2021114016_LM2_test-perplexity.txt', 'w') as f:
        f.write("Average perplexity for Witten-Bell language model is: %f\n" % avg_perplexity)
        for i,item in enumerate(perplexity):
            f.write("Sentence %d: %f\n" % (i+1, item))
              
    elif('Ulysses' in  corpus):
      with open ('./Results/2021114016_LM4_test-perplexity.txt', 'w') as f:
        f.write("Average perplexity for Witten-Bell language model is: %f\n" % avg_perplexity)
        for i,item in enumerate(perplexity):
            f.write("Sentence %d: %f\n" % (i+1, item))
      
      
    input_sentence=convert_to_sentence_tokens([input_sentence], n)
    # print(input_sentence)
    perplexity = find_perplexity1(n, input_sentence, log_probabilities_kneserney,1)
     
    
    trainset=convert_to_sentence_tokens(trainset, n)
    perplexity = find_perplexity(n, trainset, log_probabilities_kneserney)
    avg_perplexity = sum(perplexity)/len(perplexity)
    if ('Pride' in corpus):
      with open ('2021114016_LM2_train-perplexity.txt', 'w') as f:
        f.write("Average perplexity for Witten-Bell language model is: %f\n" % avg_perplexity)
        for i,item in enumerate(perplexity):
            f.write("Sentence %d: %f\n" % (i+1, item))
            
    elif('Ulysses' in  corpus):
      with open ('2021114016_LM4_train-perplexity.txt', 'w') as f:
        f.write("Average perplexity for Witten-Bell language model is: %f\n" % avg_perplexity)
        for i,item in enumerate(perplexity):
            f.write("Sentence %d: %f\n" % (i+1, item))


  


def kneserney_for_one(n, tuple_n):
    history = tuple_n[:(n-1)]
    if (n == 1):
        n_gram_counts[n].setdefault(tuple_n, 1)
        kneserney_probability = n_gram_counts[n][tuple_n] / \
            len(unique_n_grams[n].keys())
        kneserney_probabilities[n].update({tuple_n: kneserney_probability})
        return kneserney_probability

    if (tuple_n not in n_gram_counts[n].keys()):
        n_gram_counts[n].update({tuple_n: 1})

    if (n_gram_counts[n-1][history] == 0 or history not in n_gram_counts[n-1].keys()):
        n_gram_counts[n-1].update({history: 1})

    unique_n_grams[n-1].setdefault(history, 1)

    kneserney_probability = max(0, (n_gram_counts[n][tuple_n]-discount))/n_gram_counts[n-1][history] + (
        (discount*unique_n_grams[n-1][history])/n_gram_counts[n-1][history])*kneserney_for_one(n-1, tuple_n[1:])
    kneserney_probabilities[n].update({tuple_n: kneserney_probability})

    return kneserney_probability


def kneserney(n):
    for tuplee in n_gram_counts[n].keys():
        kneserney_for_one(n, tuplee)


def calculate_log_probability_kneser(n):
    for i in range(1, 5):
        for key in kneserney_probabilities[i].keys():
            log_probabilities_kneserney[i].update(
                {key: math.log(kneserney_probabilities[i][key])})


def find_perplexity1(n, sentences_tokens, log_probabilities, check=0):
    all_perplexity = []
    for sentence in sentences_tokens:
        # print("------SENTENCE------\n", sentence)
        single_perplex = 0
        for i in range(len(sentence) - n + 1):
            if tuple(sentence[i:i+n]) in log_probabilities[n].keys():
                single_perplex += log_probabilities[n][tuple(sentence[i:i+n])]
            else:

                log_probabilities[n][tuple(
                    sentence[i:i+n])] = math.log(kneserney_for_one(n, tuple(sentence[i:i+n])))
                single_perplex += log_probabilities[4][tuple(sentence[i:i+n])]
      
        if(check):
          print("Probability : ",math.exp(single_perplex))
        single_perplex /= (len(sentence) - n + 1)
        single_perplex *= -1
        single_perplex = math.exp(single_perplex)
        all_perplexity.append(single_perplex)
    return all_perplexity


def wrapper_kneserNey(n, corpus, input_sentence):
    """
    Creates a Kneser-Ney language model
    """
    # print("> Creating Kneser-Ney language model...")
    trainset, testset = split_dataset(corpus)
    vocab = create_vocab(trainset)
    common_data(n, trainset, vocab)
    # print("> Initializing Kneser-Ney language model...")
    kneserney(4)
    # print("> Done creating Kneser-Ney language model.")

    # print("> Calculating log probabilities for Kneser-Ney language model...")
    calculate_log_probability_kneser(n)
    # print("> Done calculating log probabilities for Kneser-Ney language model.")

    # print("> Calculating perplexity for Kneser-Ney language model...")
    testset = convert_to_sentence_tokens(testset, n)
    perplexity = find_perplexity1(n, testset, log_probabilities_kneserney)
    # print("> Done calculating perplexity for Kneser-Ney language model.")
    # print("> Calculating average perplexity for Kneser-Ney language model...")
    # print(perplexity)
    avg_perplexity = sum(perplexity)/len(perplexity)
    # print("> Done calculating average perplexity for Kneser-Ney language model.")
    # print("Average perplexity for Kneser-Ney language model is: ", avg_perplexity)
    print("Average perplexity : ",avg_perplexity)
    
    if ('Pride' in corpus):
      with open ('./Results/2021114016_LM1_test-perplexity.txt', 'w') as f:
        f.write("Average perplexity for Kneser-Ney language model is: %f\n" % avg_perplexity)
        for i,item in enumerate(perplexity):
            f.write("Sentence %d: %f\n" % (i+1, item))
              
    elif('Ulysses' in  corpus):
      with open ('./Results/2021114016_LM3_test-perplexity.txt', 'w') as f:
        f.write("Average perplexity for Kneser-Ney language model is: %f\n" % avg_perplexity)
        for i,item in enumerate(perplexity):
            f.write("Sentence %d: %f\n" % (i+1, item))
      
      
      
    
    trainset=convert_to_sentence_tokens(trainset, n)
    perplexity = find_perplexity1(n, trainset, log_probabilities_kneserney)
    avg_perplexity = sum(perplexity)/len(perplexity)
    if ('Pride' in corpus):
      with open ('./Results/2021114016_LM1_train-perplexity.txt', 'w') as f:
        f.write("Average perplexity for Kneser-Ney language model is: %f\n" % avg_perplexity)
        for i,item in enumerate(perplexity):
            f.write("Sentence %d: %f\n" % (i+1, item))
            
    elif('Ulysses' in  corpus):
      with open ('./Results/2021114016_LM2_train-perplexity.txt', 'w') as f:
        f.write("Average perplexity for Kneser-Ney language model is: %f\n" % avg_perplexity)
        for i,item in enumerate(perplexity):
            f.write("Sentence %d: %f\n" % (i+1, item))

    input_sentence=convert_to_sentence_tokens([input_sentence], n)
    # print(input_sentence)
    perplexity = find_perplexity1(n, input_sentence, log_probabilities_kneserney,1)
    # avg_perplexity = sum(perplexity)/len(perplexity)
    # print(avg_perplexity)
    # print(math.exp(avg_perplexity))

def language_model(n, smoothing, corpus, input_sentence):
    """
    Creates a language model
    """
    if (smoothing == 'w' or smoothing == 'W' or smoothing == 'wb' or smoothing == 'WB' or smoothing == 'WittenBell' or smoothing == 'WITTENBELL' or smoothing == 'wittenbell'):
        wrapper_wittenBell(n, corpus, input_sentence)

    if (smoothing == 'k' or smoothing == 'K' or smoothing == 'kn' or smoothing == 'KN' or smoothing == 'KneserNey' or smoothing == 'KNESERNEY' or smoothing == 'kneserney'):
        wrapper_kneserNey(n, corpus, input_sentence)


# Driver code
smoothing = sys.argv[1]
corpus = sys.argv[2]

input_sentence = input("Enter a sentence: ")

language_model(4, smoothing, corpus, input_sentence)
