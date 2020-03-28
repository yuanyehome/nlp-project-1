import numpy as np


class utils:
    data = None

    def __init__(self, raw_data, word2idx):
        """
        raw_data: [[word1, word2, ...], ...]
        word2idx: a dict
        """
        self.data = np.zeros([len(raw_data), len(word2idx)])
        for (i, item) in enumerate(raw_data):
            for word in item[1]:
                self.data[i][word2idx[word]] += 1

    def PCA(self, in_data):
        pass

    def Tf_idf(self, in_data):
        pass

    def naive_select(self, in_data, K=15000):
        sum_res = self.data.sum(axis=0)
        idxs = np.argpartition(sum_res, -K)[-K:]
        idxs = np.sort(idxs)
        return in_data[:, idxs]
