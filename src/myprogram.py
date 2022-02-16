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


class MyModel(tf.keras.Model):
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self, input_dim, output_dim, dense_dim, chars_to_id):
        super().__init__(self)
        self.lstm = LSTM(128, input_shape=(input_dim, output_dim))
        self.dropout = Dropout(0.1)
        self.dense = Dense(dense_dim, activation = 'softmax')
        self.epochs = 10
        self.seq_len = 100
        self.bs = 64
        self.chars_to_id = chars_to_id

    def call(self, inputs):
        x = inputs
        x = self.lstm(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        return x

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
        self.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
        self.fit(data_X, data_Y, batch_size = self.bs, epochs = self.epochs, verbose = 2)
    
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

    def write_pred(preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def save(self, work_dir):
        # your code here
        path = os.path.join(work_dir, 'trained_model')
        self.save(path)
        
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            # f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        path = os.path.join(work_dir, 'trained_model')
        return tf.keras.models.load_model(path)
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
        pred = model.run_pred(model, test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
