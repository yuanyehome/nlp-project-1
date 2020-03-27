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
labels = []
data_arr = []
label_arr = []
acc = []
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
        data.append(tfidf[idx] / np.linalg.norm(tfidf[idx]))
        labels.append(all_data[idx][0])


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
    for item in tqdm(all_data, desc="processing vector:"):
        arr = np.zeros(idx)
        for word in item[1]:
            arr[word2idx[word]] = 1
        data.append(arr / np.linalg.norm(arr))
        labels.append(item[0])


def divide_data():
    """
    divide the data into 10 parts
    """
    np.random.shuffle(data)
    length = len(data) // 10
    for idx in range(9):
        data_arr.append(data[idx * length:(idx + 1) * length])
        label_arr.append(labels[idx * length:(idx + 1) * length])
    data_arr.append(data[9 * length:])
    label_arr.append(labels[9 * length:])


def run_test(idx):
    """
    Use data_arr[i] as test set and the left data as train set.
    """

    # get test data and train data
    test_data = data_arr[i]
    test_labels = label_arr[i]
    train_data = []
    train_labels = []
    all_num = len(test_data)
    acc_num = 0
    for j in range(10):
        if j != idx:
            train_data += data_arr[j]
            train_labels += label_arr[j]
    test_data = np.array(test_data)
    train_data = np.array(train_data)
    test_labels = np.array(test_labels)
    train_labels = np.array(train_labels)
    res_data = np.matmul(test_data, train_data.T)
    print(res_data.shape)
    pred_labels = []

    for res in res_data:
        idxs = np.argpartition(res, -K)[-K:]
        K_labels = map(lambda id_:train_labels[id_], idxs)
        label_dict = {}
        Max = 0
        pred = None
        for label in K_labels:
            label_dict.setdefault(label, 0)
            label_dict[label] += 1
            if label_dict[label] > Max:
                Max = label_dict[label]
                pred = label
        pred_labels.append(pred)
    acc_num = np.sum(pred_labels == test_labels)

    # for each item in test set, find the k-nearest-neighbor.
    acc.append(acc_num / all_num)
    print("accuracy in test%d : %f" % (i, acc_num / all_num))
    if save_result:
        with open("../res/res-" + time.strftime("%Y-%m-%d-%H:%M:%S",
                                                time.localtime()) + ".txt", "w") as res_f:
            print("accuracy in test%d : %f" % (i, acc_num / all_num), file=res_f)


if __name__ == "__main__":
    build_data()
    divide_data()
    for i in range(10):
        run_test(i)
    print("average accuracy: %f" % (np.mean(acc)))
    if save_result:
        with open("../res/res-" + time.strftime("%Y-%m-%d-%H:%M:%S",
                                                time.localtime()) + ".txt", "w") as res_f:
            print("average accuracy: %f" % (np.mean(acc)), file=res_f)
