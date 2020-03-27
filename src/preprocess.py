import pickle
import os

all_data = []
data_path = "../data/new_weibo_13638"
labels = os.listdir(data_path)

for label in labels:
    label_path = os.path.join(data_path, label)
    for item in os.listdir(label_path):
        file_name = os.path.join(label_path, item)
        with open(file_name) as f:
            contents = f.read().split()
            if len(contents) != 0:
                data_item = (label, contents)
                all_data.append(data_item)

with open("../data/all_data.pkl", "wb") as f:
    pickle.dump(all_data, f)
