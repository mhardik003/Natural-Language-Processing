from cbow_utils import *
# import glove embeddings
import gensim.downloader

NUM_EPOCHS = 1
word_list = ['horror', 'good', 'kill', 'hero', 'action']
encoder = Word2Vec('../Dataset/Movies_and_TV_10.json')
encoder.train(NUM_EPOCHS)

for word in word_list:
    closest_k_words = encoder.get_closest(word, 10)
    closest_k_words.style.set_caption(f'Closest Words to {word}')
    print("-"*50, "\n")
    print("Closest words to ", word, " : ")
    print(closest_k_words)
    print()


encoder.plot_embeddings(closest_k_words)

closest_k_words = encoder.get_closest('titanic', 10)
print("Closest words to ", 'titanic given by our model', " : ")
print(closest_k_words)


glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
sims = glove_vectors.most_similar('titanic', topn=10)
print("Closest words to ", 'titanic given by gensim', " : ")
print(sims)


encoder.save_model()
