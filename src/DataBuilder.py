import numpy as np
import pickle
from tqdm import tqdm


class DataBuilder:
    data = None
    labels = []
    all_data = None
    data_arr = []
    label_arr = []
    raw_data_arr = []
    vocab_len = None
    passage_len = None
    word2idx = {}
    idx2word = {}

    def __init__(self, path):
        with open(path, "rb") as f:
            self.all_data = pickle.load(f)
        self.passage_len = len(self.all_data)

    def build_vocab(self):
        """
        Build vocabulary
        """
        idx = 0
        for item in self.all_data:
            for word in item[1]:
                if word not in self.word2idx.keys():
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1
            self.labels.append(item[0])
        self.vocab_len = idx
        self.labels = np.array(self.labels)

    def build_data(self):
        """
        Build data vectors by words' appearing.
        """
        assert(self.data == None)
        print("Generating text vectors")
        self.data = np.zeros([self.passage_len, self.vocab_len])
        for (i, item) in enumerate(self.all_data):
            for word in item[1]:
                self.data[i][self.word2idx[word]] = 1
        print("Done!")

    def build_data_2(self):
        """
        Build data vectors using the words' frequency.
        """
        assert(self.data == None)
        print("Generating text vectors")
        self.data = np.zeros(self.passage_len, self.vocab_len)
        for (i, item) in enumerate(self.all_data):
            for word in item[1]:
                self.data[i][self.word2idx[word]] += 1
        print("Done!")

    def normalize_data(self):
        """
        Normalize each text vector
        """
        print("Normalizing data ...")
        self.data = np.array(
            list(map(lambda item: item / np.linalg.norm(item), self.data)))
        print("Done!")

    def divide_data(self):
        """
        Divide the data into 10 parts
        """
        print("Splitting data ...")
        zip_data = list(zip(self.data, self.labels, self.all_data))
        np.random.shuffle(zip_data)
        self.data, self.labels, self.all_data = list(zip(*zip_data))
        self.data = list(self.data)
        self.labels = list(self.labels)
        self.all_data = list(self.all_data)

        length = len(self.data) // 10
        for idx in range(9):
            self.data_arr.append(self.data[idx * length:(idx + 1) * length])
            self.label_arr.append(self.labels[idx * length:(idx + 1) * length])
            self.raw_data_arr.append(
                self.all_data[idx * length:(idx + 1) * length])
        self.data_arr.append(self.data[9 * length:])
        self.label_arr.append(self.labels[9 * length:])
        self.raw_data_arr.append(self.all_data[9 * length:])
        print("Done!")
