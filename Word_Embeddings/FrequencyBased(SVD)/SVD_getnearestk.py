import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gensim.downloader



class K_nearest_words:
    def __init__(self, word, k):
        self.embeddings = None
        self.word2idx = None
        self.idx2word = None
        self.reviews_list = None
        self.vocab = None
        self.word = word
        self.k = k
        self.closest_words = None

    def load_data(self):
        """
        load the data from the pickle files
        """

        with open("./Model_Data/embeddings.pt", "rb") as f:
            self.embeddings = pickle.load(f)

        with open("./Model_Data/word2idx.pt", "rb") as f:
            self.word2idx = pickle.load(f)

        with open("./Model_Data/idx2word.pt", "rb") as f:
            self.idx2word = pickle.load(f)

        with open("./Model_Data/reviews_list.pt", "rb") as f:
            self.reviews_list = pickle.load(f)

        with open("./Model_Data/vocab.pt", "rb") as f:
            self.vocab = pickle.load(f)

    def get_predictions(self):
        """
        get the top k most similar words to the word
        """
        
        # check if the word is in the vocabulary
        if word not in self.word2idx.keys():
            print("Word not in vocabulary")
            exit()

        # get the index of the word
        word_idx = self.word2idx[word]

        # get the word embedding
        word_embedding = self.embeddings[word_idx]

        # get the cosine similarity between the word embedding and all the other embeddings
        similarities = self.embeddings.dot(word_embedding)

        # get the top k most similar words
        self.closest_words = [self.idx2word[idx]
                              for idx in similarities.argsort()[-k:][::-1] if idx != word_idx]

        closest_words_tuple = [(word, round(similarities[self.word2idx[word]], 6))
                               for word in self.closest_words]

        return closest_words_tuple

    def plot_predictions(self):
        """
        print the top k most similar words in the graph with the word in the center using t-SNE
        """

        tsne = TSNE(n_components=2, random_state=0,
                    perplexity=self.k-2, n_iter=1000)

        # get the t-SNE coordinates of the top k most similar words
        np.set_printoptions(suppress=True)

        # get the embeddings of the top k most similar words
        closest_word_embeddings = self.embeddings[[
            self.word2idx[word] for word in self.closest_words]]
        Y = tsne.fit_transform(closest_word_embeddings)

        # plot the t-SNE coordinates of the top k most similar words
        x_coords = Y[:, 0]
        y_coords = Y[:, 1]

        # plot the t-SNE coordinates of the top k most similar words
        plt.figure(figsize=(10, 5))
        plt.scatter(x_coords, y_coords)

        # add the word in the center
        plt.scatter(0, 0, c='red', s=100)
        plt.annotate(word, xy=(0, 0), color='red',
                     xytext=(0, 0), textcoords='offset points')

        # add the words as labels to the points
        for label, x, y in zip(self.closest_words, x_coords, y_coords):
            plt.annotate(label, xy=(x, y), xytext=(
                0, 0), textcoords='offset points')

        plt.title(
            't-SNE visualization of the top 10 similar words for {0}'.format(word))
        plt.savefig(
            './Model_Data/Graphs/{0}_{1}Nearest.png'.format(word, k))
        plt.show()


word = input("Enter a word : ")
k = int(input("Enter the number of closest words to find : "))
word = word.lower()
knn = K_nearest_words(word, k)
knn.load_data()
closest_words_tuple = knn.get_predictions()
knn.plot_predictions()
print(word, " : ", closest_words_tuple)


glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
sims = glove_vectors.most_similar(word, topn=10)
print("Closest words to ", 'titanic given by gensim', " : ")
print(sims)
