import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from hyperParameters import params
from dataGen import my_dataloader, my_dataset
from nn_utils import build_idx_data, build_labels, gen_split, get_parameters
import time
import sys
from networks import DNN, LSTM

# open some log files
class_num = 9
log_file = open("./log/log_%s.txt" %
                (time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())), "w")
pkl_log_file = open("./log/log_%s.pkl" %
                    (time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())), "wb")


class Net(nn.Module):
    """
    A simple feedforward neural network.
    Use nn.Embedding as a embedding layer.
    No pre-trained word vector.
    """

    def __init__(self, cfg):
        super(Net, self).__init__()
        self.embed = nn.Embedding(params['vocab_size'], params['embed_dim'])
        embed_dim = params['embed_dim']
        dropout = params['dropout']
        self.fc = nn.Linear(embed_dim, class_num)
        self.dropout = nn.Dropout(dropout)
        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        x = F.adaptive_avg_pool2d(x.unsqueeze(
            1), (1, params['embed_dim'])).squeeze()
        if self.cfg == 'dropout':
            x = self.dropout(x)
        ret = self.fc(x)
        return ret


def run_test(model, test_data):
    """
    Generate a test dataloader and run a test.
    """
    test_loader = my_dataloader(test_data)
    test_correct = 0
    test_all = 0
    with torch.no_grad():
        for test_data in test_loader:
            labels, sentences = test_data
            sentences = sentences.type(torch.LongTensor).cuda()
            labels = labels.type(torch.LongTensor).cuda()
            outputs = model(sentences)
            _, predicted = torch.max(outputs, 1)
            test_all += len(labels)
            test_correct += torch.sum(predicted == labels).item()
    return test_correct / test_all


def run_train(model, train_data, test_data, epoch_num=60):
    """
    Generate train dataloader and run train.
    """
    dataloader = my_dataloader(train_data)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    train_accs = []
    test_accs = []
    epoch_loss = []
    for epoch in range(epoch_num):
        train_correct = 0
        train_all = 0
        losses = []
        # make learning rate decreate with epoch, may be helpful for convergence.
        if epoch >= 32:
            lr = 0.0005
        if epoch >= 64:
            lr = 0.0001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for i, (labels, sentences) in enumerate(dataloader):
            optimizer.zero_grad()
            sentences = sentences.type(torch.LongTensor).cuda()
            labels = labels.type(torch.LongTensor).cuda()
            outputs = model(sentences)
            _, predicted = torch.max(outputs, 1)
            train_all += len(labels)
            train_correct += torch.sum(predicted == labels).item()
            loss = criterion(outputs, labels)
            losses.append(loss)
            loss.backward()
            optimizer.step()
            # every 60 steps print a message.
            if (i + 1) % 60 == 0:
                print("epoch: %d step: %d loss: %f" % (epoch, i, loss))
        # run a test when each epoch ends.
        test_acc = run_test(model, test_data)
        print("epoch: %d train_acc: %.2f%% test_acc: %.2f%% avg_loss: %.4f" %
              (epoch, train_correct / train_all *
               100, test_acc * 100, torch.mean(loss)))
        train_accs.append(train_correct / train_all)
        test_accs.append(test_acc)
        epoch_loss.append(torch.mean(loss).item())
        # print("epoch: %d train_acc: %.2f%% test_acc: %.2f%% avg_loss: %.4f" %
        #       (epoch, train_correct / train_all *
        #        100, test_acc * 100, torch.mean(loss)),
        #       file=log_file)
    return [train_accs, test_accs, epoch_loss]


if __name__ == "__main__":
    # process command line args
    argv = sys.argv
    cfg = None
    run_epoch = 60
    if len(argv) <= 1:
        print("Usage: python src/torch_NN.py [naive/DNN/LSTM] [dropout/None]")
        exit(0)
    if len(argv) >= 3:
        if argv[2] != 'dropout':
            raise ValueError
        cfg = argv[2]
        run_epoch = 100
    assert (argv[1] in ['naive', 'DNN', 'LSTM'])
    print(argv, file=log_file)

    # prepare data and transform them to useful format
    with open("./data/all_data.pkl", "rb") as f:
        all_data = pickle.load(f)
    all_labels, label2idx, idx2label = build_labels(all_data)
    corpus, word2idx, idx2word = build_idx_data(
        all_data, max_vocab=params['vocab_size'],
        maxlen=params['pad_len'], padding=params['pad_type']
    )
    zip_data = list(zip(all_labels, corpus))
    np.random.shuffle(zip_data)
    print("All data length: %d" % len(zip_data))
    get_parameters(Net(cfg))
    get_parameters(DNN())
    get_parameters(LSTM())

    test_accs = []
    train_logs = []
    for case in range(10):
        train_data, test_data = gen_split(zip_data, case)
        print("\033[32mTest case %d\033[0m: train_len = %d, test_len = %d" %
              (case, len(train_data), len(test_data)))
        if argv[1] == 'naive':
            model = Net(cfg).cuda()
        elif argv[1] == 'DNN':
            model = DNN().cuda()
        else:
            model = LSTM().cuda()
        train_log = run_train(
            model, train_data, test_data, epoch_num=run_epoch)
        train_logs.append(train_log)
        test_acc = run_test(model, test_data)
        test_accs.append(test_acc)
        print("\033[32mTest case %d\033[0m: acc = %.2f%%" %
              (case, test_acc * 100))
        print("Test case %d: acc = %.2f%%" %
              (case, test_acc * 100), file=log_file)
    # save the log, for plot.
    pickle.dump(train_logs, pkl_log_file)
    print("Avg acc = %f" % np.mean(test_accs))
    print("Avg acc = %f" % np.mean(test_accs), file=log_file)
