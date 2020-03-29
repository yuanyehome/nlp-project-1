import numpy as np
import pickle
import matplotlib.pyplot as plt


class utils:
    data = None
    tfidf = None

    def __init__(self, raw_data, word2idx):
        """
        raw_data: [[word1, word2, ...], ...]
        word2idx: a dict
        """
        self.data = np.zeros([len(raw_data), len(word2idx)])
        for (i, item) in enumerate(raw_data):
            for word in item[1]:
                self.data[i][word2idx[word]] += 1

    def PCA(self, in_data, N=2):
        mean_data = in_data - np.mean(in_data, axis=0)
        cov = np.cov(mean_data, rowvar=False)
        eigVals, eigVects = np.linalg.eig(cov)
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[-N:]
        redEigVects = eigVects[:, eigValInd]
        lowDDataMat = meanRemoved * redEigVects
        plt.plot(lowDDataMat[:, 0], lowDDataMat[:, 1])
        plt.savefig('res/pca.png')
        plt.show()

    def get_Tf_idf(self, in_data, K=15000):
        pass

    def select_by_Tf_idf(self, in_data):
        pass

    def naive_select(self, in_data, K=15000):
        sum_res = self.data.sum(axis=0)
        idxs = np.argpartition(sum_res, -K)[-K:]
        idxs = np.sort(idxs)
        return in_data[:, idxs]


if __name__ == "__main__":
    f = open("./data/all_data.pkl", "rb")
    all_data = pickle.load(f)
    word2idx = {}
    idx = 0
    for item in all_data:
        for word in item[1]:
            if word not in word2idx.keys():
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1
    data = np.zeros([len(all_data), idx])
    for (i, item) in enumerate(all_data):
        arr = np.zeros(idx)
        for word in item[1]:
            data[i][word2idx[word]] = 1
    select = utils(all_data, word2idx)
    select.PCA(data)
