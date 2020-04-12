import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def read_file(path):
    with open(path, "rb") as f:
        train_logs = pickle.load(f)
    return train_logs


def plot_curve(x, y, x_label, y_label, path):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path)
    plt.clf()


def plot_double_curve(x, y1, y2, legend1, legend2, x_label, y_label, path):
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend([legend1, legend2])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path)
    plt.clf()


def plot_log(x, x_label, train_log, test_log, epoch_loss, base_path):
    plot_double_curve(x, train_log, test_log, "train acc", "test acc",
                      x_label, "accuracy", base_path+"_acc.png")
    plot_curve(x, epoch_loss, x_label, 'loss',
               base_path+"_loss.png")


def draw_log(name):
    pkl_path = "./log/%s.pkl" % name
    base_path = './figures/%s' % name
    train_accs, test_accs, epoch_loss = read_file(pkl_path)[0]
    assert (len(train_accs) == len(test_accs)
            and len(test_accs) == len(epoch_loss))
    length = len(train_accs)
    x_line = np.linspace(0, length - 1, length)
    x_label = "epoch"
    plot_log(x_line, x_label, train_accs, test_accs,
             epoch_loss, base_path)


if __name__ == "__main__":
    all_log = ['simple', 'simple_dropout', 'deep_nn', 'LSTM']
    for name in all_log:
        draw_log(name)
