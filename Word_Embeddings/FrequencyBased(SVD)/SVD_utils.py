import json
from collections import Counter
import numpy as np
import string
import scipy as sp
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm
from nltk.corpus import stopwords


class COMatrixEncoder:
    """
    This class is used to create the embeddings using the co-occurrence matrix and SVD
    """

    def __init__(self,filename, NUM_REVIEWS=10000, WINDOW_SIZE=4, FREQ_UNDER_WHICH_TO_IGNORE=4):
        self.NUM_REVIEWS = NUM_REVIEWS
        self.WINDOW_SIZE = WINDOW_SIZE
        self.FREQ_UNDER_WHICH_TO_IGNORE = FREQ_UNDER_WHICH_TO_IGNORE
        # self.NUM_MOST_COMMON_WORDS_TO_IGNORE=NUM_MOST_COMMON_WORDS_TO_IGNORE
        self.vocab = Counter()
        self.word2idx = {}
        self.idx2word = {}
        self.reviews_list = []
        self.matrix = None
        self.combined_plot = None
        self.filename = filename
    
        
        self.create_data()
        self.create_vocab()
        self.create_indices()

    def create_data(self):
        """
        Creates the reviews list and the vocabulary
        """

        with open(self.filename) as f:
            data = f.readlines()

        for i in range(self.NUM_REVIEWS):
            data[i] = json.loads(data[i])
            # convert to lower case
            data[i]['reviewText'] = data[i]['reviewText'].lower()

            # remove punctuation
            data[i]['reviewText'] = data[i]['reviewText'].translate(
                str.maketrans('', '', string.punctuation))
            self.reviews_list.append(data[i]['reviewText'])
            

    def create_vocab(self):
        """
        Creates the vocabulary and the reviews list with the words replaced by <UNK> if they are not in the vocabulary
        """

        self.vocab = [
            word for review in self.reviews_list for word in review.split()]

        self.vocab = Counter(self.vocab)

        # removing the least common words (words with frequency less than or equal to 5)
        words_to_remove = [k for k in self.vocab.keys(
        ) if self.vocab[k] <= self.FREQ_UNDER_WHICH_TO_IGNORE]
        num_to_remove = sum([self.vocab[word] for word in words_to_remove])

        for word in words_to_remove:
            self.vocab.pop(word)
        self.vocab.update({'<UNK>': num_to_remove})

        # converting the words removed to <UNK>
        reviews_list = [' '.join([word if word in self.vocab.keys(
        ) else '<UNK>' for word in review.split()]) for review in self.reviews_list]

        # remove the stop words from the vocab
        for word in stopwords.words('english'):
            if word in self.vocab.keys():
                self.vocab.pop(word, None)

        # remove the common words from the reviews (sadly, that remove <UNK> (that is the least common words too )
        reviews_list = [' '.join([word if word in self.vocab.keys(
        ) else '' for word in review.split()]) for review in reviews_list]

        print("\n> Vocabulary size : ", len(self.vocab.keys()))

    def create_indices(self):
        """
         getting the word to index and index to word mappings
        """

        self.word2idx = {word: i for i, word in enumerate(self.vocab.keys())}
        self.idx2word = {i: word for i, word in enumerate(self.vocab.keys())}

    def create_cooccurrence_matrix(self):
        """
        Creates the co-occurrence matrix
        """
        print("\n> Creating the co-occurrence matrix...")
        self.matrix = sp.sparse.lil_matrix(
            (len(self.vocab.keys()), len(self.vocab.keys())))

        # filling the matrix with the co-occurrence values of the words in the reviews list with the window size of "WINDOW_SIZE"
        for review in tqdm(self.reviews_list):
            for i, word in enumerate(review.split()):
                for j in range(max(0, i-self.WINDOW_SIZE), min(len(review.split()), i+self.WINDOW_SIZE)):
                    if i != j and review.split()[j] in self.vocab.keys() and word in self.vocab.keys():
                        self.matrix[self.word2idx[word], self.word2idx[review.split()[j]]
                                    ] += 1
        # print("Is the matrix sparse : ", sp.sparse.issparse(self.matrix))

    def create_embeddings(self, embedding_size):
        """
        Creates the embeddings using SVD
        """

        # generating the embeddings using SVD
        print("\n> Creating the embeddings using SVD... (It may take a while)")

        # measure time taken
        init_time = time.time()
        self.embeddings, _, _ = sp.sparse.linalg.svds(
            self.matrix, embedding_size, return_singular_vectors='u', which='LM')
        
        print("Time taken : {} seconds.\n".format(time.time() - init_time))
    
        print('-'*50)
        # print("Embeddings shape : ", self.embeddings.shape)

    def get_closest_words(self, word, k=10):
        """
        Returns the top k most similar words to the given word
        """
        word = word.lower()

        # check if the word is in the vocabulary
        if word not in self.word2idx.keys():
            return None, None, None, "Word not in vocabulary"

        # get the index of the word
        word_idx = self.word2idx[word]

        # get the word embedding
        word_embedding = self.embeddings[word_idx]

        # get the cosine similarity between the word embedding and all the other embeddings
        similarities = self.embeddings.dot(word_embedding)

        # get the top k most similar words
        closest_words = [self.idx2word[idx]
                         for idx in similarities.argsort()[-k:][::-1] if idx != word_idx]

        closest_words_tuple = [(word, round(similarities[self.word2idx[word]],6))
                               for word in closest_words]

        # print the top k most similar words in the graph with the word in the center using t-SNE
        tsne = TSNE(n_components=2, random_state=0,
                    perplexity=k-2, n_iter=1000)

        # get the t-SNE coordinates of the top k most similar words
        np.set_printoptions(suppress=True)

        # get the embeddings of the top k most similar words
        closest_word_embeddings = self.embeddings[[
            self.word2idx[word] for word in closest_words]]
        Y = tsne.fit_transform(closest_word_embeddings)

        x_coords = Y[:, 0]
        y_coords = Y[:, 1]

        return closest_words, x_coords, y_coords, closest_words_tuple
    

    def plot_embeddings(self, word_list):
        fig, axes = plt.subplots(len(word_list), figsize=(10, 30), dpi=80)

        for i, word in enumerate(word_list):
            closest_words, x_coords, y_coords, closest_words_tuple = self.get_closest_words(
                word, 10)
            if(closest_words!="Word not in vocabulary"):
                axes[i].scatter(x_coords, y_coords)
                axes[i].set_title('t-SNE visualization of the top 10 similar words for {0}'.format(word.upper()))
                axes[i].scatter(0, 0, color='red', s=100)
                axes[i].annotate(word, xy=(0, 0), color='red',
                                xytext=(0, 0), textcoords='offset points')
                for label, x, y in zip(closest_words, x_coords, y_coords):
                    axes[i].annotate(label, xy=(x, y), xytext=(
                        0, 0), textcoords='offset points')

                print(word, " : ", closest_words_tuple)

            print("-"*50, "\n")

        # show the plot
        fig.savefig('./Model_Data/Graphs/tsne_size{0}.png'.format(self.NUM_REVIEWS))
        fig.show()
        

    def save_data(self):
        """
        Saves the embeddings, the mappings and the reviews list
        """

        # save the embeddings and the mappings
        with open('./Model_Data/embeddings.pt', 'wb') as f:
            pickle.dump(self.embeddings, f)

        with open('./Model_Data/word2idx.pt', 'wb') as f:
            pickle.dump(self.word2idx, f)

        with open('./Model_Data/idx2word.pt', 'wb') as f:
            pickle.dump(self.idx2word, f)

        with open('./Model_Data/reviews_list.pt', 'wb') as f:
            pickle.dump(self.reviews_list, f)

        with open('./Model_Data/matrix.pt', 'wb') as f:
            pickle.dump(self.matrix, f)

        with open('./Model_Data/vocab.pt', 'wb') as f:
            pickle.dump(self.vocab, f)
