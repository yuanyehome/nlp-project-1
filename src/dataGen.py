from torch.utils.data import Dataset, DataLoader
from hyperParameters import params
import numpy as np
from nn_utils import build_idx_data, build_labels
import pickle


class my_dataset(Dataset):
    """
    Build pytorch dataset
    """

    def __init__(self, train_data):
        """
        train data type: (class, ndarray([idx1, idx2, ...]))
        """
        np.random.shuffle(train_data)
        self.train_data = train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        data = self.train_data[idx]
        label = data[0]
        sentence = data[1]

        return label, sentence


def my_dataloader(train_data):
    """
    Build pytorch dataloader
    """
    dataset = my_dataset(train_data)
    return DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)


if __name__ == "__main__":
    with open("./data/all_data.pkl", "rb") as f:
        all_data = pickle.load(f)
    labels, label2idx, idx2label = build_labels(all_data)
    corpus, word2idx, idx2word = build_idx_data(
        all_data, max_vocab=params['vocab_size'],
        maxlen=params['pad_len'], padding=params['pad_type']
    )
    zip_data = list(zip(labels, corpus))
    dataset = my_dataset(zip_data)
    print(dataset.__getitem__(100))
