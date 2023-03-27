import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from scipy.spatial.distance import cosine

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


with open('./Model_Data/vocab.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

with open('./Model_Data/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

with open('./Model_Data/idx2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)

with open('./Model_Data/freq.pkl', 'rb') as f:
    freq = pickle.load(f)
    
with open('./Model_Data/weights.pkl', 'rb') as f:
    weights = pickle.load(f)
    
with open('./Model_Data/N.pkl', 'rb') as f:
    N = pickle.load(f)
    
with open('./Model_Data/EMBED_SIZE.pkl', 'rb') as f:
    EMBEDDING_SIZE = pickle.load(f)
    
model = CBOW(N, EMBEDDING_SIZE)

# load the model
model.load_state_dict(torch.load('./Model_Data/model.pt'))


model.eval()

def get_embedding( word_idx):
    """
    Returns the embedding of a word
    """

    embedding_index = torch.LongTensor([word_idx])
    return model.embeddings(embedding_index).data[0]

def get_closest(_word, k):
    """
    Returns the k closest words to the given word
    """
    
    word = _word.lower()

    if word not in vocabulary:
        print(
            f"{_word} is not in vocabulary")
        return

    distances = []
    target_index = word2idx[word]
    target_embedding = get_embedding(target_index)

    for i in range(1, N):
        if i == target_index:
            continue

        temp_embedding = get_embedding(i)
        comp_word = idx2word[i]
        dist = cosine(target_embedding, temp_embedding)
        distances.append({'Word': comp_word, 'Distance': dist})

    distances = sorted(distances, key=lambda x: x['Distance'])

    return pd.DataFrame(distances[:k])


print(get_closest('action', 10))