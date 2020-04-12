import pickle
import os

all_data = []
data_path = "../data/new_weibo_13638"
labels = os.listdir(data_path)


# Preprocess data with format of [data1, data2, ...].
# Each data is organized as (label, [word1, word2, ...])
for label in labels:
    label_path = os.path.join(data_path, label)
    for item in os.listdir(label_path):
        file_name = os.path.join(label_path, item)
        with open(file_name) as f:
            contents = f.read().split()
            # ignore empty files
            if len(contents) != 0:
                data_item = (label, contents)
                all_data.append(data_item)

# save data into a binary file, which is easier to read.
with open("../data/all_data.pkl", "wb") as f:
    pickle.dump(all_data, f)
