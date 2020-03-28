import numpy


class utils:
    data = None

    def __init__(self, raw_data, word2idx):
        """
        raw_data: [[word1, word2, ...], ...]
        word2idx: a dict
        """
        self.data = np.zeros(len(raw_data), len(word2idx))
        for (i, item) in enumerate(raw_data):
            for word in item:
                self.data[i][word2idx[word]] += 1

    def PCA(in_data):
        pass

    def Tf_idf(in_data):
        pass

    def naive_select(in_data, K=10000):
        sum_res = self.data.sum(axis=0)
        idxs = np.argpartition(sum_res, -K)[-K:]
        idxs = np.sort(idxs)
        return in_data[:, idxs]
