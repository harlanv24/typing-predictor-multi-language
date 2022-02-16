#!/usr/bin/env python
from cgitb import lookup
import os
import string
import random
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dropout, Dense, LSTM
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pickle




def load_training_data():
    with open('data/test_data.txt', encoding='UTF-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    chars_to_id = dict()
    id_to_chars = dict()
    for i, c in enumerate(chars):
        chars_to_id[c] = i
        id_to_chars[i] = c

    train_X = []
    train_Y = []
    seq_len = 100
    for i in range(len(text)-seq_len):
        train_X.append([chars_to_id[c] for c in text[i:i+seq_len]])
        train_Y.append([chars_to_id[c] for c in text[i+seq_len]])
    train_X = np.reshape(train_X, (len(train_X), seq_len, 1))
    train_Y = np_utils.to_categorical(train_Y)

    return train_X, train_Y, chars_to_id

    '''
    vocab = set(text)
    print(len(vocab))
    lookup_layer = tf.keras.layers.StringLookup(vocabulary = list(text), mask_token = None)
    id_array = lookup_layer(tf.strings.unicode_split(text, input_encoding = 'UTF-8'))
    data_tf = tf.data.Dataset.from_tensor_slices(id_array)
    data_batches = data_tf.batch(seq_len+1, drop_remainder = True)
    def generate_map(batch):
        input_text = batch[:-1]
        target_text = batch[1:]
        return input_text, target_text
    data_pairs = data_batches.map(generate_map)
    data_pairs = (data_pairs.shuffle(10000).batch(bs, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
    return data_pairs
    '''


def write_pred(preds, fname):
    with open(fname, 'wt') as f:
        for p in preds:
            f.write('{}\n'.format(p))

class MyModel():
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self, input_dim, output_dim, dense_dim, chars_to_id):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(input_dim, output_dim)))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(dense_dim, activation = 'softmax'))
        self.model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
        self.epochs = 10
        self.seq_len = 100
        self.bs = 64
        self.chars_to_id = chars_to_id


    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    def run_train(self, data_X, data_Y, work_dir):
        # your code here 
        self.model.fit(data_X, data_Y, batch_size = self.bs, epochs = self.epochs, verbose = 2)
    
    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            inp_to_id = [self.chars_to_id[c] for c in inp]
            inp_to_id = np.reshape(inp_to_id, (1, self.seq_len, 1))
            # this model just predicts a random character each time
            top_guesses = [self.model.predict(inp_to_id) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here
        path = os.path.join(work_dir, 'trained_model')
        # save = tf.keras.callbacks.ModelCheckpoint(filepath = path, save_weight_only=True)
        # save(self.model)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            # f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        path = os.path.join(work_dir, 'trained_model')
        with open(path, 'rb') as f:
            pickle.load(f)
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            # dummy_save = f.read()
        # return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Loading training data')
        train_X, train_Y, chars_to_id = load_training_data()
        print('Instatiating model')
        model = MyModel(train_X.shape[1], train_X.shape[2], train_Y.shape[1], chars_to_id)
        print('Training')
        model.run_train(train_X, train_Y, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = run_pred(model, test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
