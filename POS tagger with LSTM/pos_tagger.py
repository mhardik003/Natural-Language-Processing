import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
torch.manual_seed(0)
# load idx2tag and word2idx
with open('./Encoding_Dictionaries/idx2tag.pkl', 'rb') as f:
    idx2tag = pickle.load(f)

with open('./Encoding_Dictionaries/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
    


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim  # the number of features in the hidden state h
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim)  # embedding layer

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)  # LSTM layer

        # the linear layer that maps from hidden state space to tag space-
        self.hidden2tag = nn.Linear(
            hidden_dim, tagset_size)  # fully connected layer
        self.hidden = self.init_hidden()  # initialize the hidden state

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))  # (h0, c0)

    def forward(self, sentence):
        # get the embedding of the words
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)  # pass the embedding to the LSTM layer
        # pass the output of the LSTM layer to the fully connected layer
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # get the softmax of the output of the fully connected layer
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# load the pytorch model


EMBEDDING_DIM = 128
HIDDEN_DIM = 128
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(idx2tag))
model.load_state_dict(torch.load('pos_tagger_pretrained_model.pt'))
model.to(device)

def     prepare_sequence(seq, to_idx):
    idxs = [to_idx['<unk>'] if w not in to_idx else to_idx[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long).to(device)


# use the model to to predict the tags of the sentence
sentence = input("Enter a sentence: ")
sentence_copy = sentence.split()
sentence = sentence.lower().split()
predicted =  [idx2tag[i] for i in model(
    prepare_sequence(sentence, word2idx)).argmax(1).tolist()]


for i in range(len(sentence)):
    print(sentence_copy[i],"\t", predicted[i])