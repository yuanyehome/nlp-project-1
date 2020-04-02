import numpy as np
import pickle
from tqdm import tqdm
import time
from my_utils import utils
from DataBuilder import DataBuilder
import sys
import math
from functools import reduce

# global variables：control some files and debug info.
start_time = time.time()
save_result = False
DEBUG = False
np.random.seed(0)
res_f = None
dbg_file = None
dbg_print_detail = True


def process_args():
    """
    process command line args.
    support `-save`, `-debug`, `-seed [int]` or `seed time`.
    note that `-seed` should be placed at the end of args
    """
    global save_result, DEBUG, res_f, dbg_file
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

    if DEBUG:
        dbg_file = open("./dbg/dbg-" +
                        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + ".txt", "w")


def print_detail(pred_label_arr, label_arr, res_f=None):
    """
    Print error rate of each label.
    """
    pred_labels = reduce(lambda p, q: p + q, pred_label_arr)
    std_labels = reduce(lambda p, q: p + q, label_arr)
    all_labels = np.unique(std_labels)
    label_class = {}
    corr_class = {}
    for label in all_labels:
        label_class.setdefault(label, 0)
        corr_class.setdefault(label, 0)
    for (pred, std) in zip(pred_labels, std_labels):
        label_class[std] += 1
        corr_class[std] += (pred == std)
    for label in all_labels:
        print("label: %s  num: %-4d  corr: %-4d  acc:%.2f%%" % (
            label, label_class[label], corr_class[label],
            100 * corr_class[label] / label_class[label]))
        if res_f is not None:
            print("label: %s  num: %-4d  corr: %-4d  acc:%.2f%%" % (
                label, label_class[label], corr_class[label],
                100 * corr_class[label] / label_class[label]), file=res_f)


def get_res(train_data, train_labels, test_data, test_labels, raw_data=None, K=20):
    """
    Given training set and test set, get the number of correct prediction.
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
            # score = similarity + 1
            label_dict[label] += ress[i]
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
                  (pred_labels[idx], test_labels[idx], raw_data[idx][1]), file=dbg_file)
    return acc_num, pred_labels


def search_K(train_data, train_labels):
    """
    Deprecated.
    This function is used for searching the best K given a training set.
    I divided 1/9 of training set as validation set, and search for the best K. 
    """
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


def run_test(idx, data_arr, label_arr, raw_data_arr, acc, all_pred_labels, selector):
    """
    Run the i-th test.
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
    # selected_idxs = selector.select_by_KL(train_data, train_labels, 15000)
    # train_data = train_data[:, selected_idxs]
    # test_data = test_data[:, selected_idxs]
    # Deprecated. Searching K is useless...
    # K = search_K(train_data, train_labels)

    print("Running test %d ..." % (idx))
    acc_num, pred_labels = get_res(train_data, train_labels, test_data,
                                   test_labels, raw_data_arr[i], 40)
    all_pred_labels.append(pred_labels)
    acc.append(acc_num / all_num)

    print("accuracy in test%d : %f" % (idx, acc_num / all_num))
    print("elapsed time: %s s" % (time.time() - start_time))
    if save_result:
        print("accuracy in test%d : %f" %
              (idx, acc_num / all_num), file=res_f)
        print("elapsed time: %s s" %
              (time.time() - start_time), file=res_f)


if __name__ == "__main__":
    process_args()
    path = "./data/all_data.pkl"
    db = DataBuilder(path)
    db.build_vocab()
    # db.build_data()
    select = utils(db.all_data, db.word2idx)
    # [info] Use tf-idf as text embedding.
    db.data = select.get_Tf_idf()
    # db.data = select.select_by_Tf_idf(db.data)
    # [info] get the idxs of most N frequent words
    idxs = select.naive_select(db.data, 15000)
    # [info] save the selected features.
    db.select_features(idxs, DEBUG=DEBUG, dbg_file=dbg_file)
    # [info] change the norm of each embedding to 1
    db.normalize_data()
    # [info] divide the data into 10 parts
    db.divide_data()
    acc = []
    all_pred_labels = []
    for i in range(10):
        run_test(i, db.data_arr, db.label_arr,
                 db.raw_data_arr, acc, all_pred_labels, select)
    print("average accuracy: %f" % (np.mean(acc)))
    print("total time: %s s" % (time.time() - start_time))
    if save_result:
        print("average accuracy: %f" % (np.mean(acc)), file=res_f)
        print("total time: %s s" % (time.time() - start_time), file=res_f)
    if dbg_print_detail:
        print_detail(all_pred_labels, db.label_arr, res_f)
