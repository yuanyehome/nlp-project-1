import numpy as np
import pickle
from tqdm import tqdm
import time

data = []
data_arr = []
acc = []
K = 20
save_result = False
with open("../data/all_data.pkl", "rb") as f:
    all_data = pickle.load(f)


def divide_data():
    """
    divide the data into 10 parts
    """
    np.random.shuffle(data)
    length = len(data) // 10
    for idx in range(9):
        data_arr.append(data[idx * length:(idx + 1) * length])
    data_arr.append(data[9 * length:])


def train():
    pass


def test(idx):
    pass


if __name__ == "__main__":
    divide_data()
