import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy


class utils:
    data = None
    tfidf = None
    label_class = []

    def __init__(self, raw_data, word2idx):
        """
        raw_data: [[word1, word2, ...], ...]
        word2idx: a dict
        """
        self.data = np.zeros([len(raw_data), len(word2idx)])
        for (i, item) in enumerate(raw_data):
            for word in item[1]:
                self.data[i][word2idx[word]] += 1
            self.label_class.append(item[0])
        self.label_class = np.unique(self.label_class)

    def PCA(self, in_data, labels, N=2):
        """
        Get 2d points using PCA.
        Print some other features.
        """
        # 60000的维数求协方差矩阵特征值根本算不下来……只能先选择少一点的特征再降维；
        # 这样PCA做出来的二维图好像也看不出来啥东西……
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

    def get_Tf_idf(self):
        """
        Build text vectors based on tf-idf.
        """
        print("Calculating tf-idf")
        self.tfidf = copy.deepcopy(self.data)
        passage_num = len(self.tfidf)
        word_num = self.tfidf.shape[1]
        row_sum = np.sum(self.tfidf, axis=1).reshape(
            [passage_num, 1]).repeat(word_num, axis=1)
        self.tfidf /= row_sum
        col_sum = np.log(passage_num / np.sum(self.tfidf > 0, axis=0)) \
                    .reshape([1, word_num]) \
                    .repeat(passage_num, axis=0)
        self.tfidf *= col_sum
        print("Done!")
        return self.tfidf

    def select_by_Tf_idf(self, in_data, k=2):
        assert(self.tfidf is not None)
        args = self.tfidf.argsort()
        idxs = np.array([], dtype=int)
        passage_num = len(self.tfidf)
        for i in range(passage_num):
            idxs = np.append(idxs, args[i][-k:])
        selected_idxs = np.unique(idxs)
        print("number of features selected: %d" % (len(selected_idxs)))
        return in_data[:, selected_idxs]

    def select_by_KL(self, in_data, labels, K=15000):
        word_num = in_data.shape[1]
        label_class = np.unique(labels)
        label_map = {}
        label_num = len(label_class)
        P = np.zeors(label_num)
        for i in range(label_num):
            label_map[label_class[i]] = i
        for label in labels:
            P[label_map[i]] += 1
        P /= len(labels)
        labels_arr = np.array([labels])
        Q = []
        for i in range(label_num):
            tmp = np.sum((in_data > 0) *
                         (np.repeat(labels_arr.T, word_num, axis=1) == label_class[i]), axis=0)
            Q.append(tmp)
        Q = np.array(Q)
        Q /= np.repeat(np.sum(Q), label_num, axis=0)
        P = np.repeat(np.reshape(P, [label_num, 1]), axis=0)
        ans = np.sum(Q * np.log((Q + 1e-8) / P), axis=0)

    def naive_select(self, in_data, K=15000):
        """
        Select features by the frequency of word.
        """
        sum_res = self.data.sum(axis=0)
        idxs = np.argpartition(sum_res, -K)[-K:]
        idxs = np.sort(idxs)
        self.selected_idxs = idxs
        print("feature num: %d" % (len(idxs)))
        return in_data[:, idxs]

    def chi_square(self, in_data, labels, K=15000):
        passage_num = len(in_data)
        chi_square_score = []
        for label in self.label_class:
            label_data = in_data[np.where(labels == label)]
            not_label_data = in_data[np.where(labels != label)]
            A = np.sum(label_data > 0, axis=0)
            B = np.sum(not_label_data > 0, axis=0)
            C = len(label_data) - A
            D = len(not_label_data) - B
            chi_square_score.append((A * D - B * C + 1e-7) ** 2 /
                                    ((A + B) * (C + D) + 1e-5))
        chi_square_score = np.max(chi_square_score, axis=0)
        selected_idxs = np.argsort(chi_square_score)[-K:]
        return selected_idxs


def data_insight():
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


if __name__ == "__main__":
    data_insight()
