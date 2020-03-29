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

    def PCA(self, in_data, labels, N=2):
        mean_data = in_data - np.mean(in_data, axis=0)
        cov = np.cov(mean_data, rowvar=False)
        eigVals, eigVects = np.linalg.eig(cov)
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[-N:]
        redEigVects = eigVects[:, eigValInd]
        lowDDataMat = np.matmul(mean_data, redEigVects)
        all_labels = np.unique(labels)
        print("label num: %d" % (len(all_labels)))
        for label in all_labels:
            print("number of label %s: %d" % (label, np.sum(labels == label)))
        label2idx = {}
        for i in range(len(all_labels)):
            label2idx[all_labels[i]] = i
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        plt.scatter(lowDDataMat[:, 0], lowDDataMat[:, 1], s=1, c=list(
            map(lambda label: colors[label2idx[label]], labels)))
        plt.savefig('./res/pca.png', dpi=600)
        # plt.show()

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
                idx += 1
    data = np.zeros([len(all_data), idx])
    labels = []
    for (i, item) in enumerate(all_data):
        arr = np.zeros(idx)
        for word in item[1]:
            data[i][word2idx[word]] = 1
        labels.append(item[0])
    labels = np.array(labels)
    select = utils(all_data, word2idx)
    data = select.naive_select(data, 5000)
    select.PCA(data, labels)
