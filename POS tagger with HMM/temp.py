from collections import OrderedDict, namedtuple
from pprint import pprint
# Sentence = namedtuple()
Sentence = namedtuple("Sentence", "words tags")

# read through the brown-universal text file
"""Read tagged sentence data"""
with open('brown-universal.txt', 'r') as f: 
    # return a list of lists where each inside list is an array of words of each sentence
    sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    # pprint(sentence_lines)
# return an ordered dictionary having tuples having tags and words
key_sentence_dict=OrderedDict()
for s in sentence_lines :
    if(s[0]):
        words_tags_list =[]
        for l in s[1:]:
            words_tags_list.append(l.strip().split("\t"))
            # words_tags_list.
        
        single_Sentence = Sentence(*zip(*words_tags_list))
        key_sentence_dict.update([(s[0],single_Sentence)]   )


pprint(key_sentence_dict)
# pprint(popo)