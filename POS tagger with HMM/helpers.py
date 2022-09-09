import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import networkx as nx
import random
from pprint import pprint


from io import BytesIO
from itertools import chain
from collections import namedtuple, OrderedDict


def model2png(model, filename="", overwrite=False, show_ends=False):
    """Convert a Pomegranate model into a PNG image
    The conversion pipeline extracts the underlying NetworkX graph object,
    converts it to a PyDot graph, then writes the PNG data to a bytes array,
    which can be saved as a file to disk or imported with matplotlib for display.
        Model -> NetworkX.Graph -> PyDot.Graph -> bytes -> PNG
    Parameters
    ----------
    model : Pomegranate.Model
        The model object to convert. The model must have an attribute .graph
        referencing a NetworkX.Graph instance.
    filename : string (optional)
        The PNG file will be saved to disk with this filename if one is provided.
        By default, the image file will NOT be created if a file with this name
        already exists unless overwrite=True.
    overwrite : bool (optional)
        overwrite=True allows the new PNG to overwrite the specified file if it
        already exists
    show_ends : bool (optional)
        show_ends=True will generate the PNG including the two end states from
        the Pomegranate model (which are not usually an explicit part of the graph)
    """

    nodes = model.graph.nodes()
    if not show_ends:
        nodes = [n for n in nodes if n not in (model.start, model.end)]
    g = nx.relabel_nodes(model.graph.subgraph(nodes), {
                         n: n.name for n in model.graph.nodes()})
    pydot_graph = nx.drawing.nx_pydot.to_pydot(g)
    pydot_graph.set_rankdir("LR")
    png_data = pydot_graph.create_png(prog='dot')
    img_data = BytesIO()
    img_data.write(png_data)
    img_data.seek(0)
    if filename:
        if os.path.exists(filename) and not overwrite:
            raise IOError(
                "File already exists. Use overwrite=True to replace existing files on disk.")
        with open(filename, 'wb') as f:
            f.write(img_data.read())
        img_data.seek(0)
    return mplimg.imread(img_data)


def show_model(model, figsize=(5, 5), **kwargs):
    """Display a Pomegranate model as an image using matplotlib
    Parameters
    ----------
    model : Pomegranate.Model
        The model object to convert. The model must have an attribute .graph
        referencing a NetworkX.Graph instance.
    figsize : tuple(int, int) (optional)
        A tuple specifying the dimensions of a matplotlib Figure that will
        display the converted graph
    **kwargs : dict
        The kwargs dict is passed to the model2png program, see that function
        for details
    """
    plt.figure(figsize=figsize)
    plt.imshow(model2png(model, **kwargs))
    plt.axis('off')


Sentence = namedtuple("Sentence", "words tags")


def read_data(filename):  # read through the brown-universal text file
    """Read tagged sentence data"""
    with open(filename, 'r') as f:
        # return a list of lists where each inside list is an array of words of each sentence
        # [[sentence 1 words list], [sentence 2 words list] and so on...]
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
        # pprint(sentence_lines)

    # return an ordered dictionary having tuples having tags and words
    # for s in sentence_lines if s[0] checks if the sentence does have a key associated with it
    # if yes then l will store all the words in the sentence ( removing the key from the top of each sentence)
    # we then split the the tags from the words and then use strip to remove any extra spaces left in between.
    # we then make two tuples (one of the words and the other of the tags) using zip and keep storing it in our named tuple 'Sentence'
    # and then at the end make an Ordered Dictionary with the key as the  key of the sentence and the element as the tuple

    # return an ordered dictionary having tuples having tags and words
    key_sentence_dict = OrderedDict()
    for s in sentence_lines:
        if (s[0]):
            words_tags_list = []
            for l in s[1:]:
                words_tags_list.append(l.strip().split("\t"))
                # words_tags_list.

        single_Sentence = Sentence(*zip(*words_tags_list))
        key_sentence_dict.update([(s[0], single_Sentence)])

    # pprint(key_sentence_dict)
    return (key_sentence_dict)

    # or instead of that big loop we can write this one-liner
    return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                        for l in s[1:]]))) for s in sentence_lines if s[0]))


def read_tags(filename):  # here the file name is 'tags-universal.txt'
    """Read a list of word tag classes"""
    with open(filename, 'r') as f:
        tags = f.read().split("\n")  # this file had tags seperated by  '\n'
    # frozen set is a set which is immutable (since we don't need to make any changes to this set)
    return frozenset(tags)


class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y N stream")):
    def __new__(cls, sentences, keys):
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, {k: sentences[k] for k in keys}, keys, wordset, word_sequences,
                               tagset, tag_sequences, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())


class Dataset(namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set testing_set N stream")):
    def __new__(cls, tagfile, datafile, train_test_split=0.8, seed=112890):
        tagset = read_tags(tagfile)  # file containing all the tags
        sentences = read_data(datafile)  # file containing the data
        # creating a tuple of the keys of the sentences in the dataset
        keys = tuple(sentences.keys())
        # merges all the words from all the tuples (sentence tuples)
        # sentence.values will return the values from the dictionary ( that is the namedtuple of the words and their respective tags )
        # from this tuple we are choosing the words and then chaining (combining these words to form a frozenset (immutableset))
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))

        # forms a tuple of lists where each list contains words from each sentence
        word_sequences = tuple([sentences[k].words for k in keys])

        # forms a tuple of tags where each list contains words from each sentence
        tag_sequences = tuple([sentences[k].tags for k in keys])

        # N stores the total number of distunguised words
        # s will store the named tuple of sentence words and their tags
        # from there we are choosing only the tuple of words and chaining them all in a single tuple
        # then we are iterating through this chain and adding 1 for each word to N
        # _ is used when we don't need a name for the variable that is not used ( since here we do not need the name of the words in tuple we just need to count them therefore we dont need to store it)
        N = sum(1 for _ in chain(*(s.words for s in sentences.values())))

        # creating a list out the the tuple 'keys'
        _keys = list(keys)


        # creating training and testing data
        if seed is not None:
            random.seed(seed) #random number generator
        random.shuffle(_keys) #shuffles the list of keys 
        split = int(train_test_split * len(_keys))  # as given above train test split is 0.8  (therefore the number of training data is 80% of the data)

        # next we split the _keys list into two subsets (one of training data and other of testing data)
        training_data = Subset(sentences, _keys[:split])
        testing_data = Subset(sentences, _keys[split:])

        #creating a tuple of the two tuples of words and their respective tags
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))


        # returning all this processed data
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences, tagset,
                               tag_sequences, training_data, testing_data, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())
