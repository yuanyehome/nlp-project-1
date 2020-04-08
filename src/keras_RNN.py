import pickle
from tensorflow import keras
import numpy as np
import tensorflow.keras.layers as layers
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def build_labels(data):
    labels = np.zeros([len(data), 9])
    label2idx = {}
    idx2label = {}
    idx = 0
    for item in data:
        if item[0] not in label2idx.keys():
            label2idx[item[0]] = idx
            idx2label[idx] = item[0]
            idx += 1
    for (i, item) in enumerate(data):
        labels[i][label2idx[item[0]]] = 1
    return labels, label2idx, idx2label


def build_datas(data):
    word2idx = {}
    idx2word = {}
    word2idx['<START>'] = 1
    word2idx['<PAD>'] = 0
    idx2word[1] = '<START>'
    idx2word[0] = '<PAD>'
    idx = 2
    for item in data:
        for word in item[1]:
            if word not in word2idx.keys():
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1
    vocab_len = idx - 1
    new_data = []
    for item in data:
        new_data.append([])
        this_data = new_data[-1]
        this_data.append(1)
        for word in item[1]:
            this_data.append(word2idx[word])
    new_data = np.array(new_data)
    print("vocab size: %d" % vocab_len)
    return new_data, word2idx, idx2word, vocab_len


with open("./data/all_data.pkl", "rb") as f:
    all_data = pickle.load(f)

labels, label2idx, idx2label = build_labels(all_data)
new_data, word2idx, idx2word, vocab_len = build_datas(all_data)
zip_data = list(zip(new_data, labels))
np.random.shuffle(zip_data)
new_data, labels = list(zip(*zip_data))
new_data = np.array(list(new_data))
labels = np.array(list(labels))

train_x = new_data[0:10000]
train_y = labels[0:10000]
test_x = new_data[10000:]
test_y = labels[10000:]

train_x = keras.preprocessing.sequence.pad_sequences(
    train_x, value=word2idx['<PAD>'],
    padding='post', maxlen=100
)
test_x = keras.preprocessing.sequence.pad_sequences(
    test_x, value=word2idx['<PAD>'],
    padding='post', maxlen=100
)
model = keras.Sequential()
model.add(layers.Embedding(vocab_len + 2, 32, input_length=100))
model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.1))
model.add(layers.Dense(9, activation='softmax'))
model.summary()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

x_train = train_x[:8000]
y_train = train_y[:8000]
x_val = train_x[8000:]
y_val = train_y[8000:]
history = model.fit(x_train, y_train,
                    epochs=200, batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

result = model.evaluate(test_x, test_y)
print(result)
