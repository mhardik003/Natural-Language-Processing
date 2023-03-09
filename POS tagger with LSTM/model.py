
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import pickle
import numpy as np
from torchmetrics import F1Score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

print("Using device:", device)


# load the dataset
with open('./UD_English-Atis/en_atis-ud-dev.conllu', 'r') as f:
    dev_data = f.read()

with open('./UD_English-Atis/en_atis-ud-train.conllu', 'r') as f:
    train_data = f.read()

with open('./UD_English-Atis/en_atis-ud-test.conllu', 'r') as f:
    test_data = f.read()

# print("\nTrain data size:", len(train_data))
# print("Dev data size:", len(dev_data))
# print("Test data size:", len(test_data))
# print("Train data sample:", train_data[:1000])


def get_sentences(data):
    """ Function to get sentences from the dataset """

    sentences = []
    for line in data.split('\n'):
        if line.startswith('# text = '):
            sentences.append(line[9:])

    return sentences


train_sentences = get_sentences(train_data)
test_sentences = get_sentences(test_data)
dev_sentences = get_sentences(dev_data)

# print ("\nTrain sentences size:", len(train_sentences))
# print("Test sentences size:", len(train_sentences))
# print("Dev sentences size:", len(dev_sentences))


def get_labels(data):
    labels = []
    words = []
    output_data = []

    for line in data.split('\n'):
        if (line):
            if line.startswith('# text = ') or line.startswith('# sent_id = '):
                continue

            # getting the first 4 elements of the line
            temp = line.split('\t')[0:4]
            temp = temp[0:2]+[temp[3]]  # removing the 3rd element of the line

            # temp = index, word, label

            if temp[0] == '1':
                words = []
                labels = []
                output_data.append((words, labels))

            words.append(temp[1])
            labels.append(temp[2])

    return output_data


training_data = get_labels(train_data)
testing_data = get_labels(test_data)
dev_data = get_labels(dev_data)


# print(training_data)


word2idx = {}
tags2idx = {}
idx2tag = {}
i=0
for sent, tags in training_data:
    for word in sent:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    
    if(i%30==0):
        # choose a random number between 0 and length of the sentence
        rand = np.random.randint(0, len(sent))
        training_data[i][0][rand] = '<unk>'   # to add unknown tags to the training data
        # print("Random number:", rand)
        # print(training_data[i][0])

    for tag in tags:
        if tag not in tags2idx:
            tags2idx[tag] = len(tags2idx)
            idx2tag[len(tags2idx)-1] = tag
            
    i+=1
    

word2idx['<unk>'] = len(word2idx)
tags2idx['<unk>'] = len(tags2idx)
idx2tag[len(tags2idx)-1] = '<unk>'

# save the dictionaries
with open('./Encoding_Dictionaries/word2idx.pkl', 'wb') as f:
    pickle.dump(word2idx, f)


with open('./Encoding_Dictionaries/idx2tag.pkl', 'wb') as f:
    pickle.dump(idx2tag, f)

# print("\nVocab size:", len(word2idx))
# print("Tags size:", len(tags2idx))
# print("Tags:", tags2idx)
# print("Vocab:", word2idx)


# for embed_size in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    # for output_size in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:

        # if(embed_size!=output_size):
            # continue
# for learning_rate in [0.25, 0.01, 0.3, 0.1, 0.5, 0.15, 0.4, 0.02, 0.05, 0.2, 0.08, 0.001]:
    # for epoch_ in [5, 10, 20]:
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
# the learning rate which is used to update the weights
LEARNING_RATE = 0.2
NUM_EPOCHS = 10  # the number of times the model is trained on the entire dataset

print("Embedding size:", EMBEDDING_DIM)
print("Output size:", HIDDEN_DIM)
print("Learning rate:", LEARNING_RATE)
print("Epochs:", NUM_EPOCHS)



class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim  # the number of features in thhidden state h
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim)  # embedding layer

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)  # LSTM layer

        # the linear layer that maps from hidden state space to tag spacembed_sizee-
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

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(
    word2idx), len(tags2idx)).to(device)  # initialize the model

loss_function = nn.NLLLoss()  # define the loss function

# define the optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

def prepare_sequence(seq, to_idx):
    idxs = [to_idx['<unk>'] if w not in to_idx else to_idx[w]
            for w in seq]
    return torch.tensor(idxs, dtype=torch.long).to(device)

# Training the model and evaluating it on the test set

def train_model(model, data, num_epoch):

    for epoch in range(num_epoch):
        for sentence, tags in data:
            model.zero_grad()  # clear the gradients of all optimized variables
            model.hidden = model.init_hidden()  # initialize the hidden state

            # convert the sentence to a tensor
            sentence_in = prepare_sequence(sentence, word2idx)
            # convert the tags to a tensor
            targets = prepare_sequence(tags, tags2idx)

            tag_scores = model(sentence_in)  # forward pass
            # calculate the loss
            loss = loss_function(tag_scores, targets)
            loss.backward()  # backward pass
            optimizer.step()  # update the parameters

        # print("Epoch:", epoch, "Loss:", loss.item())

train_model(model, training_data, NUM_EPOCHS)

# save the model

model_path = 'pos_tagger_pretrained_model.pt'

torch.save(model.state_dict(), model_path)

def evaluate(model, data):
    model.eval()
    correct = 0
    total = 0
    for sentence, tags in data:
        sentence_in = prepare_sequence(sentence, word2idx)
        targets = prepare_sequence(tags, tags2idx)
        tag_scores = model(sentence_in)
        _, predicted = torch.max(tag_scores, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    return 100 * correct / total

print("\nAccuracy on the training set:",
        evaluate(model, training_data))
print("Accuracy on the dev set:", evaluate(model, dev_data))
print("Accuracy on the test set:", evaluate(model, testing_data))

# Computing the F1 score of the model
f1 = F1Score(task="multiclass", num_classes=len(tags2idx))
f1.to(device)
actual = []
pred = []

for sentence, tags in testing_data  :
    sentence_in = prepare_sequence(sentence, word2idx)
    targets = prepare_sequence(tags, tags2idx)
    tag_scores = model(sentence_in)
    _, predicted = torch.max(tag_scores, 1)
    actual.extend(targets.tolist())
    pred.extend(predicted.tolist())

f1.update(torch.tensor(pred), torch.tensor(actual))
f1.compute()
print("F1 score:", f1.compute().item()*100)

print("\n")
sentence = "i would like the cheapest flight from pittsburgh to atlanta leaving april twenty fifth and returning may sixth"
sentence = sentence.lower().split()
sentence_copy = sentence
predicted = [idx2tag[i] for i in model(
    prepare_sequence(sentence, word2idx)).argmax(1).tolist()]

print("Sentence:", sentence_copy)
print("Predicted:", predicted)

print("=================================================================================================\n")



 
(52, 32, 15) #number of words in a sentence, number of batches, number of tags