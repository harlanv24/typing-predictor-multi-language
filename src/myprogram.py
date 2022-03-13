#!/usr/bin/env python
import os
import random
import tensorflow as tf
import numpy as np
import pickle
from langdetect import detect
from keras.optimizers import *
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dropout, Dense, LSTM
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# dictionary of language -> model here, to be accessed for predictions once input language is detected
lstm_dim = 128
epochs = 30
dropout = 0.1
seq_len = 7
bs = 64
threshold = 10
chars_to_id = dict()
id_to_chars = dict()
language_set = ['zh', 'es','en', 'hi', 'ar', 'pt', 'bn', 'ru', 'ja', 'fr']

def load_training_data(language):
    # loop through languages, store model for every language
    with open(f'data/{language}/mergedfiles.txt', encoding='UTF-8') as f:
        text = f.read().replace('\n',' ')
    counts = {}
    for char in text:
        if char in counts:
            counts[char] += 1
        else:
            counts[char] = 1
    unique_chars = []
    for k, v in counts.items():
        if v > threshold:
            unique_chars.append(k)
    chars = sorted(unique_chars)

    for i, c in enumerate(chars):
        chars_to_id[c] = i
        id_to_chars[i] = c
    chars_to_id['<unk>'] = -2
    id_to_chars[-2] = '<unk>'
    train_X = []
    train_Y = []
    
    with open(f'data/{language}/mergedfiles2.txt', encoding='UTF-8') as f:
        for text in f:
            text = text.strip()
            for i in range(min(seq_len-1, len(text)-1)):
                temp = [[chars_to_id[c] if c in chars_to_id else chars_to_id['<unk>'] for c in text[:i+1]]]
                train_X.append(tf.keras.preprocessing.sequence.pad_sequences(temp, maxlen = seq_len, padding='pre', value=-1)[0])
                train_Y.append([chars_to_id[c] if c in chars_to_id else chars_to_id['<unk>'] for c in text[i+1]])
            for i in range(len(text)-seq_len):
                train_X.append([chars_to_id[c] if c in chars_to_id else chars_to_id['<unk>'] for c in text[i:i+seq_len]])
                train_Y.append([chars_to_id[c] if c in chars_to_id else chars_to_id['<unk>'] for c in text[i+seq_len]])
        train_X = np.reshape(train_X, (len(train_X), seq_len, 1))
        train_Y = np_utils.to_categorical(train_Y)
    return train_X, train_Y

def run_pred(model, data):
    # your code here
    print(id_to_chars)
    preds = []
    right = 0
    for inp in data:
        inp, correct = inp.split("\t")
        # check for language type here
        temp = []
        inp_to_id = [chars_to_id[c] if c in chars_to_id else chars_to_id['<unk>'] for c in inp]
        temp.append(inp_to_id)
        padded_ids = tf.keras.preprocessing.sequence.pad_sequences(temp, maxlen = seq_len, padding='pre', value=-1)[0]
        inp_to_id = np.reshape(padded_ids, (1, seq_len, 1))
        top_guesses = model.predict(inp_to_id)
        sorted_guesses = sorted(enumerate(top_guesses[0]), key = lambda e:  e[1], reverse=True)
        top_3 = [id_to_chars[c[0]] for c in sorted_guesses[:3]]
        preds.append(''.join(top_3))
        right += 1 if correct in top_3 else 0
    print(right/len(preds))
    return preds

def write_pred(preds, fname):
    with open(fname, 'wt', encoding='UTF-8') as f:
        for p in preds:
            f.write('{}\n'.format(p))

class MyModel():
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self, input_dim, output_dim, dense_dim, language):
        self.model = Sequential()
        self.model.add(LSTM(lstm_dim, input_shape=(input_dim, output_dim)))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(dense_dim, activation = 'softmax'))
        self.model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
        self.language = language
        self.vocab = None

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
        self.model.fit(data_X, data_Y, batch_size = bs, epochs = epochs, verbose = 2)
    
    def save(self, work_dir):
        # your code here
        path = os.path.join(work_dir, f'trained_model_{self.language}')
        # save = tf.keras.callbacks.ModelCheckpoint(filepath = path, save_weight_only=True)
        # save(self.model)
        self.model.save(path)

        with open(path + f"/chars_to_id_dict_{self.language}.pkl", "wb") as pfile:
            pickle.dump([chars_to_id, id_to_chars], pfile)
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            # f.write('dummy save')

    @classmethod
    def load(cls, work_dir, language):
        # your code here
        global chars_to_id, id_to_chars
        path = os.path.join(work_dir, 'trained_model')
        with open(path + f"/chars_to_id_dict_{language}.pkl",  "rb") as pfile:
            chars_to_id, id_to_chars = pickle.load(pfile)
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
    parser.add_argument('--language', help='which language to train', default='English')

    args = parser.parse_args()
    language = args.language

    random.seed(0)

    if args.mode == 'train':
        def model_train_activity(language):
            train_X, train_Y = load_training_data(language)
            print(f'Instatiating model for {language}')
            model = MyModel(train_X.shape[1], train_X.shape[2], train_Y.shape[1], language)
            print(f'Training for {language}')
            model.run_train(train_X, train_Y, args.work_dir)
            print(f'Saving model {language}')
            model.save(args.work_dir)

        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Loading training data')
        if language == 'all':
            for l in set(df['1']):
                model_train_activity(l)
        else:
            model_train_activity(language)
    elif args.mode == 'test':
        print(f'Loading model for {language}')
        model = MyModel.load(args.work_dir, language)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = run_pred(model, test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
