import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

hps = {
    'lr': 1e-3,
    'hidden_size': 256,
    'batch_size': 256,
    'layer_num': 3,
    'class_num': 9,
    'cell_type': 'lstm',
    'pad_size': 100,
    'epoch_num': 100
}
tf.get_logger().setLevel('ERROR')


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


def lstm_cell(cell_type, num_nodes, keep_prob):
    if cell_type == 'lstm':
        cell = tf.contrib.rnn.BasicLSTMCell(num_nodes)
    else:
        cell = tf.contrib.rnn.LSTMBlockCell(num_nodes)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


class dataGen:
    x_data = None
    y_data = None

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def gen_batch(self):
        zip_data = list(zip(self.x_data, self.y_data))
        np.random.shuffle(zip_data)
        self.x_data, self.y_data = list(zip(*zip_data))
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        res = []
        idx = 0
        while idx < self.x_data.shape[0] - hps['batch_size']:
            res.append([self.x_data[idx:idx+hps['batch_size']],
                        self.y_data[idx:idx+hps['batch_size']]])
            idx += hps['batch_size']
        res.append([self.x_data[idx:], self.y_data[idx:]])
        return res


if __name__ == "__main__":
    # preprocess data
    with open("../data/all_data.pkl", "rb") as f:
        all_data = pickle.load(f)
    labels, label2idx, idx2label = build_labels(all_data)
    corpus = []
    for item in all_data:
        corpus.append(' '.join(item[1]))
    tokenizer = Tokenizer(15000)
    tokenizer.fit_on_texts(corpus)
    vocab = tokenizer.word_index
    assert (0 not in vocab.values())
    x_train, x_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.1)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = pad_sequences(x_train, maxlen=hps['pad_size'])
    x_test = pad_sequences(x_test, maxlen=hps['pad_size'])
    train_gen = dataGen(x_train, y_train)
    test_gen = dataGen(x_test, y_test)

    X_input = tf.placeholder(tf.int32, [None, hps['pad_size']])
    Y_input = tf.placeholder(tf.float32, [None, hps['class_num']])
    keep_prob = tf.placeholder(tf.float32, [])
    X = tf.one_hot(X_input, 15000, axis=-1)

    mlstm_cell = tf.contrib.rnn.MultiRNNCell([
        lstm_cell(hps['cell_type'], hps['hidden_size'], keep_prob=keep_prob)
        for _ in range(hps['layer_num'])],
        state_is_tuple=True
    )

    init_state = mlstm_cell.zero_state(hps['batch_size'], dtype=tf.float32)
    outputs = list()
    state = init_state
    with tf.variable_scope('RNN'):
        for timestep in range(hps['pad_size']):
            (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
            outputs.append(cell_output)
    h_state = outputs[-1]

    W = tf.Variable(tf.truncated_normal([hps['hidden_size'], hps['class_num']], stddev=0.1),
                    dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[hps['class_num']]), dtype=tf.float32)
    y_pred = tf.nn.softmax(tf.matmul(h_state, W) + bias)

    loss = -tf.reduce_mean(Y_input * tf.log(y_pred))
    train_op = tf.train.AdamOptimizer(hps['lr']).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hps['epoch_num']):
            batches = train_gen.gen_batch()
            costs = []
            accs = []
            for (X_batch, Y_batch) in batches:
                cost, acc, _ = sess.run([loss, accuracy, train_op],
                                        feed_dict={
                                            X_input: X_batch,
                                            Y_input: Y_batch,
                                            keep_prob: 0.5
                                        })
                costs.append(cost)
                accs.append(acc)
            test_cost, test_acc, _ = sess.run([loss, accuracy, train_op],
                                              feed_dict={
                                                  X_input: x_test,
                                                  Y_input: y_test,
                                                  keep_prob: 1.0
                                              })
            print("epoch %d info: cost = %.4f, acc = %.4f, test_cost = %.4f, test_acc = %.4f"
                  % (epoch, float(np.mean(costs)), float(np.mean(accs)), test_cost, test_acc))
