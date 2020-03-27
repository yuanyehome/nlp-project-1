import numpy as np
import pickle
from tqdm import tqdm
import time
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Variable `data` contains all the data item with format of (label, text embedding, |embedding|^2). The embedding is
# a vector of |V| dimension, where |V| is the size of vocabulary. The i-th element of this vector is 1 of this text
# contains word_i, else 0.
# Divide the data into 10 parts, we can get `data_arr`
# Variable `acc` records the accuracy of each test.
# Variable `res_f` is the file which the result will be written to.
# `K` is the hyperparameter.
data = []
data_arr = []
acc = []
res_f = open("../res/res-" + time.strftime("%Y-%m-%d-%H:%M:%S",
                                           time.localtime()) + ".txt", "w")
K = 20
save_result = False
with open("../data/all_data.pkl", "rb") as f:
    all_data = pickle.load(f)


def calc_cos(arr1, arr2):
    """
    Calculate the cos similarity
    """
    # assert (len(arr1) == len(arr2))
    inner = 0
    for id1 in arr1[1]:
        if id1 in arr2[1]:
            inner += 1
    return inner / math.sqrt(arr1[2] * arr2[2])


def build_data_from_sklearn():
    """
    Build the dataset and divide it into 10 parts.
    """
    global data
    corpus = list(map(lambda item: " ".join(item[1]), all_data))
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus)).toarray()
    length = len(all_data)
    for idx in range(length):
        data.append([all_data[idx][0], tfidf[idx], np.linalg.norm(tfidf[idx])])


def build_data():
    """
    Build the dataset and divide it into 10 parts.
    """
    word2idx = {}
    idx = 0
    global data
    global data_arr

    # build a index, which map a word into a index of embedding vector.
    for item in all_data:
        for word in item[1]:
            if word not in word2idx.keys():
                word2idx[word] = idx
                idx += 1

    # build vectors.
    for item in all_data:
        data_item = [item[0], [], 0]
        for word in item[1]:
            if word2idx[word] not in data_item[1]:
                data_item[1].append(word2idx[word])
                data_item[2] += 1
        data.append(data_item)


def divide_data():
    """
    divide the data into 10 parts
    """
    np.random.shuffle(data)
    length = len(data) // 10
    for idx in range(9):
        data_arr.append(data[idx * length:(idx + 1) * length])
    data_arr.append(data[9 * length:])


def run_test(idx):
    """
    Use data_arr[i] as test set and the left data as train set.
    """

    # get test data and train data
    test_data = data_arr[i]
    train_data = []
    all_num = len(test_data)
    acc_num = 0
    for j in range(10):
        if j != idx:
            train_data += data_arr[j]

    # for each item in test set, find the k-nearest-neighbor.
    for item in tqdm(test_data):
        labels = {}

        # for each item in train set, calculate cos similarity of test item and train item
        # record contains item formatted as (label, similarity).
        record = list(map(lambda train_item: (train_item[0], calc_cos(item, train_item)),
                          train_data))

        # find the max-k labels
        record.sort(key=lambda s: s[1], reverse=True)
        for m in range(K):
            labels.setdefault(record[m][0], 0)
            labels[record[m][0]] += 1

        # find the most frequent label
        res = ''
        Max = 0
        for neighbor in labels.keys():
            if labels[neighbor] > Max:
                res = neighbor
                Max = labels[neighbor]
        if res == item[0]:
            acc_num += 1
    acc.append(acc_num / all_num)
    print("accuracy in test%d : %f" % (i, acc_num / all_num))
    if save_result:
        print("accuracy in test%d : %f" % (i, acc_num / all_num), file=res_f)


if __name__ == "__main__":
    build_data()
    divide_data()
    for i in range(10):
        run_test(i)
    print("average accuracy: %f" % (np.mean(acc)))
    if save_result:
        print("average accuracy: %f" % (np.mean(acc)), file=res_f)
