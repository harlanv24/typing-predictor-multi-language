#!/usr/bin/env python
from cgitb import lookup
import os
import string
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

input_dim = 100
output_dim = 256
lstm_dim = 100
epochs = 5
dropout = 0.1
seq_len = 100
bs = 64

class MyModel():
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self):
        self.model = Sequential()
        self.model.add(Embedding(input_dim, output_dim))
        self.model.add(LSTM(lstm_dim, dropout = dropout))
        self.model.add(Dense(2, activation = 'softmax'))
        self.model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
        self.vocab = None
    
    @classmethod
    def load_training_data(cls):
        with open('data/test_data.txt') as f:
            text = f.read().decode(encoding = 'utf-8')
        return text, sorted(set(text))

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here 
        self.vocab = sorted(set(data))
        print(len(self.vocab))
        lookup_layer = tf.keras.layers.StringLookup(vocabulary = list(self.vocab), mask_token = None)
        id_array = lookup_layer(tf.strings.unicode_split(data, input_encoding = 'UTF-8'))
        data_tf = tf.data.Dataset.from_tensor_slices(id_array)
        data_batches = data_tf.batch(seq_len+1, drop_remainder = True)
        data_pairs = map()
        for batch in data_batches:
            input = batch[:-1]
            output = batch[1:]
            data_pairs[input] = output
        data_pairs = (data_pairs.shuffle(10000).batch(bs, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
        self.model.fit(data_pairs, epochs = epochs, verbose = 2)
        

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            lookup_layer = tf.keras.layers.StringLookup(vocabulary = list(self.vocab), mask_token = None)
            inp_ids = lookup_layer(inp)
            top_guesses = [self.model(inp_ids) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here
        path = os.path.join(work_dir, 'trained_model')
        save = tf.keras.callbacks.ModelCheckpoint(filepath = path, save_weight_only=True)
        save(self)
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            # f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        path = os.path.join(work_dir, 'trained_model')
        return tf.saved_model.load(path)
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
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
