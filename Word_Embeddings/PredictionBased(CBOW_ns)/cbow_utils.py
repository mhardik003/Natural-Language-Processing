from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from scipy.spatial.distance import cosine
import pandas as pd
import pickle
from collections import Counter
import os
import json
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        input_embeds = self.embeddings(inputs)
        embeds = torch.mean(input_embeds, dim=1)
        out = self.linear(embeds)
        return F.log_softmax(out, dim=1)


class Word2Vec:
    def __init__(self, filename, NUM_REVIEWS=10000, WINDOW_SIZE=4, EMBEDDING_SIZE=100, FREQ_UNDER_WHICH_TO_IGNORE=4, NEG_SAMPLE_SIZE=10, LEARNING_RATE=0.0001):

        self.filename = filename
        self.NUM_REVIEWS = NUM_REVIEWS
        self.reviews = self.get_data()
        self.FREQ_UNDER_WHICH_TO_IGNORE = FREQ_UNDER_WHICH_TO_IGNORE
        self.oov_token = '<OOV>'
        self.WINDOW_SIZE = WINDOW_SIZE
        self.EMBEDDING_SIZE = EMBEDDING_SIZE

        self.vocabulary = {self.oov_token}
        self.word2idx = {self.oov_token: 0}
        self.idx2word = {0: self.oov_token}

        self.freq = Counter()
        self.freq_dist = [0]
        self.total_word_count = 0

        self.build_vocabulary()

        self.BATCH_SIZE = 64
        self.NEG_SAMPLE_SIZE = NEG_SAMPLE_SIZE

        self.model = CBOW(self.N, self.EMBEDDING_SIZE)
        self.dataset = self.build_dataset()
        self.weights = self.negative_sampling()
        self.optimizer = optim.Adam(self.model.parameters(), LEARNING_RATE)

    def get_data(self):
        """
        Returns the reviews list
        """

        if os.path.exists('./Model_Data/parsed_reviews.pkl'):
            with open('./Model_Data/parsed_reviews.pkl', 'rb') as f:
                temp = pickle.load(f)
                if len(temp) == self.NUM_REVIEWS:
                    print("Reviews loaded from previous parsed file")
                    reviews = temp

                else:
                    with open(self.filename, 'r') as f:
                        data = f.readlines()

                    reviews = []
                    print("Parsing Reviews: ")
                    for r in tqdm(data[:self.NUM_REVIEWS]):
                        reviews.append(Review(json.loads(r)))
                    with open('./Model_Data/parsed_reviews.pkl', 'wb') as f:
                        pickle.dump(reviews, f)

        else:
            reviews = []
            print("Parsing Reviews: ")
            for r in tqdm(data[:self.NUM_REVIEWS]):
                reviews.append(Review(json.loads(r)))
            with open('./Model_Data/parsed_reviews.pkl', 'wb') as f:
                pickle.dump(reviews, f)

        return reviews

    def build_vocabulary(self):
        """
        Builds the vocabulary and word2idx and idx2word dictionaries
        """

        self.freq = Counter(
            [word for review in self.reviews for word in review])

        index = 1
        for token, f in self.freq.items():
            if f > self.FREQ_UNDER_WHICH_TO_IGNORE:
                self.vocabulary.add(token)
                self.freq_dist.append(f)
                index += 1
            else:
                self.freq_dist[0] += f

        self.word2idx = {token: index for index,
                         token in enumerate(self.vocabulary)}
        self.idx2word = {index: token for index,
                         token in enumerate(self.vocabulary)}

        self.total_word_count = sum(self.freq.values())
        self.N = len(self.vocabulary)
        print(f"Total Vocabulary Size: {self.N}")

        # remove words from reviews which are not in reviews
        for i in range(len(self.reviews)):
            self.reviews[i] = [token for token in self.reviews[i]
                               if token in self.vocabulary]

    def build_dataset(self):
        """
        Builds the dataset for training
        """

        print("Building Dataset")
        dataset = []
        for review in tqdm(self.reviews):
            for i in range(self.WINDOW_SIZE, len(review) - self.WINDOW_SIZE):
                target = review[i]
                target_index = self.word2idx[target]
                context_indices = []
                for j in range(i - self.WINDOW_SIZE, i + self.WINDOW_SIZE + 1):
                    if i == j:
                        continue
                    context = review[j]
                    context_index = self.word2idx[context]
                    context_indices.append(context_index)
                dataset.append((context_indices, target_index))

        return dataset

    def negative_sampling(self):
        """
        Computes the weights for negative sampling
        """

        print("\nComputing Weights for Negative Sampling")
        normalized_freq = F.normalize(
            torch.Tensor(self.freq_dist).pow(0.75), dim=0)  # p(w)^0.75
        weights = torch.ones(len(self.freq_dist))  # weights for each word

        for _ in tqdm(range(len(self.freq_dist))):
            for _ in range(self.NEG_SAMPLE_SIZE):
                neg_index = torch.multinomial(normalized_freq, 1)[
                    0]  # sample a word
                # increase the weight of the sampled word
                weights[neg_index] += 1

        return weights

    def train(self, num_epochs):
        """
        Trains the model
        """

        losses = []
        loss_fn = nn.NLLLoss(weight=self.weights)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}")
            total_loss = 0
            for i in tqdm(range(0, len(self.dataset), self.BATCH_SIZE)):
                batch = self.dataset[i: i + self.BATCH_SIZE]

                context = torch.LongTensor([context for context, _ in batch])
                target = torch.LongTensor([target for _, target in batch])

                self.optimizer.zero_grad()
                log_probs = self.model(context)
                loss = loss_fn(log_probs, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            print(f"Loss: {loss.item()}")
            losses.append(total_loss)

    def get_embedding(self, word_idx):
        """
        Returns the embedding of a word
        """

        embedding_index = torch.LongTensor([word_idx])
        return self.model.embeddings(embedding_index).data[0]

    def get_closest(self, _word, k):
        """
        Returns the k closest words to the given word
        """
        
        word = _word.lower()

        if word not in self.vocabulary:
            print(
                f"{_word} is not in vocabulary")
            return

        distances = []
        target_index = self.word2idx[word]
        target_embedding = self.get_embedding(target_index)

        for i in range(1, self.N):
            if i == target_index:
                continue

            temp_embedding = self.get_embedding(i)
            comp_word = self.idx2word[i]
            dist = cosine(target_embedding, temp_embedding)
            distances.append({'Word': comp_word, 'Distance': dist})

        distances = sorted(distances, key=lambda x: x['Distance'])

        return pd.DataFrame(distances[:k])

    def plot_embeddings(self, words):
        """
        Plots the embeddings of the given words
        """

        for word in words:
            if word not in self.vocabulary:
                print(
                    f"{word} is not in vocabulary")
                return

        embeddings = []
        for word in words:
            target_index = self.word2idx[word]
            target_embedding = self.get_embedding(target_index)
            embeddings.append(target_embedding)

        embeddings = torch.stack(embeddings)
        tsne = TSNE(n_components=2)
        embeddings = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 10))
        for i, word in enumerate(words):
            plt.scatter(embeddings[i, 0], embeddings[i, 1])
            plt.annotate(word, (embeddings[i, 0], embeddings[i, 1]))
        plt.show()
        plt.savefig(f"./Model_Data/Graphs/{word}.png")

    def save_model(self):
        """
        Saves the model and the vocabulary
        """
        
        torch.save(self.model.state_dict(), "./Model_Data/model.pt")

        # save vocabulary as pickle file
        with open('./Model_Data/vocab.pkl', 'wb') as f:
            pickle.dump(self.vocabulary, f)

        with open('./Model_Data/word2idx.pkl', 'wb') as f:
            pickle.dump(self.word2idx, f)

        with open('./Model_Data/idx2word.pkl', 'wb') as f:
            pickle.dump(self.idx2word, f)

        with open('./Model_Data/freq.pkl', 'wb') as f:
            pickle.dump(self.freq, f)
            
        with open('./Model_Data/weights.pkl', 'wb') as f:
            pickle.dump(self.weights, f)
            
        with open('./Model_Data/N.pkl', 'wb') as f:
            pickle.dump(self.N, f)
            
        with open('./Model_Data/EMBED_SIZE.pkl', 'wb') as f:
            pickle.dump(self.EMBEDDING_SIZE, f)
