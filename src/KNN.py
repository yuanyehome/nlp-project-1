import numpy as np
import pickle
from tqdm import tqdm
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Variable `data` contains all the data item with format of (label, text embedding, |embedding|^2). The embedding is
# a vector of |V| dimension, where |V| is the size of vocabulary. The i-th element of this vector is 1 of this text
# contains word_i, else 0.
# Divide the data into 10 parts, we can get `data_arr`
# Variable `acc` records the accuracy of each test.
# Variable `res_f` is the file which the result will be written to.
# `K` is the hyperparameter.
start_time = time.time()
data = []
labels = []
data_arr = []
label_arr = []
raw_data_arr = []
acc = []
K = 20
vocab_len = 0
word2idx = {}
idx2word = {}
save_result = False
DEBUG = False
np.random.seed(0)
with open("./data/all_data.pkl", "rb") as f:
    all_data = pickle.load(f)
if save_result:
    res_f = open("./res/res-" + time.strftime("%Y-%m-%d-%H:%M:%S",
                                              time.localtime()) + ".txt", "w")
else:
    res_f = None

if DEBUG:
    dbg_file = open("dbg-" + time.strftime("%Y-%m-%d-%H:%M:%S",
                                           time.localtime()) + ".txt", "w")
else:
    dbg_file = None


def get_res(train_data, train_labels, test_data, test_labels, raw_data=None):
    """
    get the number of correct prediction
    """
    res_data = np.matmul(test_data, train_data.T)
    pred_labels = []
    all_top_K = []

    for res in res_data:
        idxs = np.argpartition(res, -K)[-K:]
        K_labels = list(map(lambda id_: train_labels[id_], idxs))
        all_top_K.append(K_labels)
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
    if raw_data and DEBUG:
        false_idxs = np.where((pred_labels == test_labels) == False)[0]
        for idx in false_idxs:
            print("pred: %s    real: %s    text: %s" %
                  (pred_labels[idx], test_labels[idx], raw_data[idx][1]))
    return acc_num


def build_data():
    """
    Build the dataset and divide it into 10 parts.
    """
    idx = 0
    global data
    global labels
    global word2idx
    global idx2word
    global vocab_len

    # build a index, which map a word into a index of embedding vector.
    for item in all_data:
        for word in item[1]:
            if word not in word2idx.keys():
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1
    vocab_len = idx
    # build vectors.
    for item in tqdm(all_data, desc="processing passage vector"):
        arr = np.zeros(idx)
        for word in item[1]:
            arr[word2idx[word]] = 1
        data.append(arr / np.linalg.norm(arr))
        labels.append(item[0])
    data = np.array(data)
    labels = np.array(labels)


def divide_data():
    """
    divide the data into 10 parts
    """
    global data, labels, all_data
    zip_data = list(zip(data, labels, all_data))
    np.random.shuffle(zip_data)
    data, labels, all_data = list(zip(*zip_data))
    data = list(data)
    labels = list(labels)
    all_data = list(all_data)

    length = len(data) // 10
    for idx in range(9):
        data_arr.append(data[idx * length:(idx + 1) * length])
        label_arr.append(labels[idx * length:(idx + 1) * length])
        raw_data_arr.append(all_data[idx * length:(idx + 1) * length])
    data_arr.append(data[9 * length:])
    label_arr.append(labels[9 * length:])
    raw_data_arr.append(all_data[9 * length:])


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

    acc_num = get_res(train_data, train_labels, test_data,
                      test_labels, raw_data_arr[i])
    acc.append(acc_num / all_num)

    print("accuracy in test%d : %f" % (idx, acc_num / all_num))
    print("elapsed time: %s s" % (time.time() - start_time))
    if save_result:
        print("accuracy in test%d : %f" %
              (idx, acc_num / all_num), file=res_f)
        print("elapsed time: %s s" %
              (time.time() - start_time), file=res_f)


if __name__ == "__main__":
    build_data()
    divide_data()
    for i in range(10):
        run_test(i)

    print("average accuracy: %f" % (np.mean(acc)))
    print("total time: %s s" % (time.time() - start_time))
    if save_result:
        print("average accuracy: %f" % (np.mean(acc)), file=res_f)
        print("total time: %s s" % (time.time() - start_time), file=res_f)
