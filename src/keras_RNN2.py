from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import one_hot

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

with open("./data/all_data.pkl", "rb") as f:
    all_data = pickle.load(f)


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


labels, label2idx, idx2label = build_labels(all_data)
corpus = []
for item in all_data:
    corpus.append(' '.join(item[1]))

tokenizer = Tokenizer(15000)
tokenizer.fit_on_texts(corpus)
vocab = tokenizer.word_index
x_train, x_test, y_train, y_test = train_test_split(
    corpus, labels, test_size=0.1)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(15000, 128),
    tf.keras.layers.LSTM(
        128, dropout=0.2, recurrent_dropout=0.1, return_sequences=True),
    tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=['accuracy'])

# print(model.summary())
history = model.fit(x_train, y_train, epochs=100,
                    batch_size=512, verbose=1, validation_split=0.15)
model.evaluate(x_test, y_test)
