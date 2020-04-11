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
from nn_utils import build_idx_data, build_labels
import time


class_num = 9
log_file = open("./log/log_%s.txt" %
                (time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())), "w")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = nn.Embedding(params['vocab_size'], params['embed_dim'])
        embed_dim = params['embed_dim']
        dropout = params['dropout']
        self.fc = nn.Linear(embed_dim, class_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embed(x)
        x = F.adaptive_avg_pool2d(x.unsqueeze(
            1), (1, params['embed_dim'])).squeeze()
        ret = self.fc(x)
        return ret


def run_test(model, test_data):
    test_loader = my_dataloader(test_data)
    test_correct = 0
    test_all = 0
    with torch.no_grad():
        for test_data in test_loader:
            labels, sentences = test_data
            sentences = sentences.type(torch.LongTensor)
            outputs = model(sentences)
            _, predicted = torch.max(outputs, 1)
            test_all += len(labels)
            test_correct += torch.sum(predicted == labels).item()
    return test_correct / test_all


def run_train(model, train_data, epoch_num=100):
    dataloader = my_dataloader(train_data)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    for epoch in range(100):
        train_correct = 0
        train_all = 0
        losses = []
        if epoch >= 30:
            lr = 0.0005
        if epoch >= 60:
            lr = 0.0001
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        for i, (labels, sentences) in enumerate(dataloader):
            optimizer.zero_grad()
            sentences = sentences.type(torch.LongTensor)
            labels = labels.type(torch.LongTensor)
            outputs = net(sentences)
            _, predicted = torch.max(outputs, 1)
            train_all += len(labels)
            train_correct += torch.sum(predicted == labels).item()
            loss = criterion(outputs, labels)
            losses.append(loss)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("epoch: %d step: %d loss: %f" % (epoch, i, loss))
        test_acc = run_test(net, test_data)
        print("epoch: %d train_acc: %.2f%% test_acc: %.2f%% avg_loss: %.4f" %
              (epoch, train_correct / train_all *
               100, test_acc * 100, torch.mean(loss)),
              file=log_file)


if __name__ == "__main__":
    with open("./data/all_data.pkl", "rb") as f:
        all_data = pickle.load(f)
    all_labels, label2idx, idx2label = build_labels(all_data)
    corpus, word2idx, idx2word = build_idx_data(
        all_data, max_vocab=params['vocab_size'],
        maxlen=params['pad_len'], padding=params['pad_type']
    )
    zip_data = list(zip(all_labels, corpus))
    np.random.shuffle(zip_data)
    print("data length: %d" % len(zip_data))
    train_data = zip_data[0:-len(zip_data) // 10]
    test_data = zip_data[-len(zip_data) // 10:]
    print("split done!")

    net = Net()
    print(net)
    run_train(net, train_data)
