import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from hyperParameters import params


def build_labels(data):
    labels = np.zeros(len(data))
    label2idx = {}
    idx2label = {}
    idx = 0
    for item in data:
        if item[0] not in label2idx.keys():
            label2idx[item[0]] = idx
            idx2label[idx] = item[0]
            idx += 1
    for (i, item) in enumerate(data):
        labels[i] = label2idx[item[0]]
    return labels, label2idx, idx2label


def build_idx_data(data, maxlen=200, max_vocab=15000, padding='post'):
    assert (padding in ['pre', 'post'])
    corpus = []
    for item in data:
        corpus.append(' '.join(item[1]))
    tokenizer = Tokenizer(max_vocab)
    tokenizer.fit_on_texts(corpus)
    word2idx = tokenizer.word_index
    idx2word = {}
    for item in word2idx:
        idx2word[word2idx[item]] = item
    assert (0 not in word2idx.values())
    corpus = tokenizer.texts_to_sequences(corpus)
    corpus = pad_sequences(corpus, maxlen=maxlen, padding=padding)
    return corpus, word2idx, idx2word


if __name__ == "__main__":
    with open("./data/all_data.pkl", "rb") as f:
        all_data = pickle.load(f)
    labels, label2idx, idx2label = build_labels(all_data)
    corpus, word2idx, idx2word = build_idx_data(
        all_data, max_vocab=params['vocab_size'],
        maxlen=params['pad_len'], padding=params['pad_type']
    )
    with open("./data/new_data.pkl", "wb") as f:
        pickle.dump([[labels, label2idx, idx2label],
                     [corpus, word2idx, idx2word]], f)
    print("done")
