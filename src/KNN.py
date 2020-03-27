import numpy as np
import pickle
from tqdm import tqdm
import time

word2idx = {}
idx = 0
data = []
K = 20


def calc_cos(arr1, arr2):
    assert (len(arr1) == len(arr2))
    return np.inner(arr1, arr2) / np.sqrt(np.inner(arr1, arr1) * np.inner(arr2, arr2))


with open("../data/all_data.pkl", "rb") as f:
    all_data = pickle.load(f)

for item in all_data:
    for word in item[1]:
        if word not in word2idx.keys():
            word2idx[word] = idx
            idx += 1

for item in all_data:
    data_item = (item[0], np.zeros(len(word2idx)))
    for word in item[1]:
        data_item[1][word2idx[word]] = 1
    data.append(data_item)

np.random.shuffle(data)
data_arr = []
length = len(data) // 10
for i in range(9):
    data_arr.append(data[i * length:(i + 1) * length])
data_arr.append(data[9 * length:])

acc = []
res_f = open("../res/res" + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()), "w")
for i in range(10):
    test_data = data_arr[i]
    train_data = []
    all_num = len(test_data)
    acc_num = 0
    for j in range(10):
        if j != i:
            train_data += data_arr[j]
    for item in tqdm(test_data):
        labels = {}
        record = list(map(lambda train_item: (train_item[0], calc_cos(item[1], train_item[1])),
                          train_data))
        record.sort(key=lambda s: s[1], reverse=True)
        for m in range(K):
            labels.setdefault(record[m][0], 0)
            labels[record[m][0]] += 1
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
    print("accuracy in test%d : %f" % (i, acc_num / all_num), file=res_f)
print("average accuracy: %f" % (np.mean(acc)))
print("average accuracy: %f" % (np.mean(acc)), file=res_f)
