import numpy as np
import pickle
from tqdm import tqdm
import time
from my_utils import utils
import sys
import math

# Variable `data` contains all the data item with format of (label, text embedding, |embedding|^2). The embedding is
# a vector of |V| dimension, where |V| is the size of vocabulary. The i-th element of this vector is 1 of this text
# contains word_i, else 0.
# Divide the data into 10 parts, we can get `data_arr`
# Variable `acc` records the accuracy of each test.
# Variable `res_f` is the file which the result will be written to.
start_time = time.time()
data = []
labels = []
data_arr = []
label_arr = []
raw_data_arr = []
acc = []
vocab_len = 0
word2idx = {}
idx2word = {}
save_result = False
DEBUG = False
np.random.seed(0)
if "-save" in sys.argv:
    save_result = True
if "-debug" in sys.argv:
    DEBUG = True
if "-seed" in sys.argv:
    if sys.argv[-1] == 'time':
        np.random.seed(int(time.time()))
    else:
        np.random.seed(int(sys.argv[-1]))
with open("./data/all_data.pkl", "rb") as f:
    all_data = pickle.load(f)
if save_result:
    res_f = open("./res/res-" + time.strftime("%Y-%m-%d-%H:%M:%S",
                                              time.localtime()) + ".txt", "w")
else:
    res_f = None

if DEBUG:
    dbg_file = open("./dbg/dbg-" + time.strftime("%Y-%m-%d-%H:%M:%S",
                                                 time.localtime()) + ".txt", "w")
else:
    dbg_file = None


def get_res(train_data, train_labels, test_data, test_labels, raw_data=None, K=20):
    """
    get the number of correct prediction
    """
    res_data = np.matmul(test_data, train_data.T)
    pred_labels = []
    all_top_K = []

    for res in res_data:
        idxs = np.argpartition(res, -K)[-K:]
        ress = res[idxs]
        tmp = list(zip(idxs, ress))
        tmp.sort(key=lambda item: item[1], reverse=True)
        idxs, ress = list(zip(*tmp))
        idxs = list(idxs)
        K_labels = train_labels[idxs]
        all_top_K.append(list(zip(K_labels, ress)))
        label_dict = {}
        Max = 0
        pred = None
        for (i, label) in enumerate(K_labels):
            if ress[i] == 0:
                continue
            label_dict.setdefault(label, 0)
            label_dict[label] += ress[i]
            label_dict[label] += 1
            # +1 + ress[i]目前效果最好
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
            print("pred: %s    real: %s    text: %s" %
                  (pred_labels[idx], test_labels[idx], raw_data[idx][1]), file=dbg_file)
    return acc_num


def search_K(train_data, train_labels):
    print("Searching for the best K ...")
    best_K = 10
    best_acc = 0
    length = len(train_data)
    tr_data = train_data[0:8 * (length // 9)]
    val_data = train_data[8 * (length // 9):]
    tr_labels = train_labels[0:8 * (length // 9)]
    val_labels = train_labels[8 * (length // 9):]
    all_num = len(val_data)
    for i in range(100):
        acc_num = get_res(tr_data, tr_labels, val_data, val_labels, K=i+10)
        print("K: %d    accuracy: %f" % (i + 10, acc_num / all_num))
        if acc_num > best_acc:
            best_acc = acc_num
            best_K = i + 10
            print("Best K update")
    print("Search done, the best K is %d" % (best_K))
    return best_K


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
        data.append(arr)
        labels.append(item[0])
    data = np.array(data)
    labels = np.array(labels)


def build_data_2():
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
            arr[word2idx[word]] += 1
        data.append(arr)
        labels.append(item[0])
    data = np.array(data)
    labels = np.array(labels)


def normalize_data():
    global data
    print("Normalizing data ...")
    data = np.array(list(map(lambda item: item / np.linalg.norm(item), data)))
    print("Done!")


def divide_data():
    """
    divide the data into 10 parts
    """
    print("Splitting data ...")
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
    print("Done!")


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
    # K = search_K(train_data, train_labels)

    print("Running test %d ..." % (idx))
    acc_num = get_res(train_data, train_labels, test_data,
                      test_labels, raw_data_arr[i], K=15)
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
    select = utils(all_data, word2idx)
    data = select.naive_select(data)
    normalize_data()
    divide_data()
    for i in range(10):
        run_test(i)

    print("average accuracy: %f" % (np.mean(acc)))
    print("total time: %s s" % (time.time() - start_time))
    if save_result:
        print("average accuracy: %f" % (np.mean(acc)), file=res_f)
        print("total time: %s s" % (time.time() - start_time), file=res_f)
