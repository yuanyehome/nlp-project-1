import numpy as np
import pickle
from tqdm import tqdm
import time
from my_utils import utils
from DataBuilder import DataBuilder
import sys
import math


start_time = time.time()
save_result = False
DEBUG = False
np.random.seed(0)


def process_args():
    if "-save" in sys.argv:
        save_result = True
    if "-debug" in sys.argv:
        DEBUG = True
    if "-seed" in sys.argv:
        if sys.argv[-1] == 'time':
            np.random.seed(int(time.time()))
        else:
            np.random.seed(int(sys.argv[-1]))
    if save_result:
        res_f = open("./res/res-" +
                     time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + ".txt", "w")
    else:
        res_f = None

    if DEBUG:
        dbg_file = open("./dbg/dbg-" +
                        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + ".txt", "w")
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


def run_test(idx, data_arr, label_arr, raw_data_arr, acc):
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
    # process_args()
    path = "./data/all_data.pkl"
    db = DataBuilder(path)
    db.build_vocab()
    db.build_data()
    select = utils(db.all_data, db.word2idx)
    # data = select.get_Tf_idf()
    db.data = select.naive_select(db.data)
    db.normalize_data()
    db.divide_data()
    acc = []
    for i in range(10):
        run_test(i, db.data_arr, db.label_arr, db.raw_data_arr, acc)

    print("average accuracy: %f" % (np.mean(acc)))
    print("total time: %s s" % (time.time() - start_time))
    if save_result:
        print("average accuracy: %f" % (np.mean(acc)), file=res_f)
        print("total time: %s s" % (time.time() - start_time), file=res_f)
