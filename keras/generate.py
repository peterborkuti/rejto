'''
#Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import tensorflow as tf

D_GPU = '/gpu:0'
D_CPU = '/cpu:0'


class Util:
    all_characters = string.printable
    n_characters = len(all_characters)

    @staticmethod
    def read_file(filename):
        file = open(filename).read()

        return file, len(file)

    # Turning a string into a tensor

    @staticmethod
    def code_string(string):
        coded_list = [0] * len(string)
        for c in range(len(string)):
            try:
                coded_list[c]= Util.all_characters.index(string[c])
            except ValueError:
                print("Character is not printable!")  
                
        return coded_list

    @staticmethod
    def char_tensor(string):
        return Variable(torch.LongTensor(Util.code_string(string)))
    
    @staticmethod
    def decode_char_tensor(tensor):
        decoded_list = [' '] * len(tensor)
        for c in range(len(tensor)):
            decoded_list[c] = Util.all_characters[tensor[c].item()]
            
        return "".join(decoded_list)

    # Readable time elapsed

    @staticmethod
    def time_since(since):
        s = time.time() - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

class CharGenerator:
    # path = get_file(
    #    'nitzsche.txt',
    #    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    # with io.open(path, encoding='utf-8') as f:
    #    text = f.read().lower()

    chars = set()
    indices_char = dict()
    char_indices = dict()

    oh_sentences = None
    oh_next_chars = None

    model = None

    maxlen = 40

    def __init__(self, path='../books/unidecoded_rejto.txt', text_size=20000):
        with tf.device(D_CPU):
            text = open(path).read(text_size)
            print('corpus length:', len(text))

            chars = sorted(list(set(text)))
            print('total chars:', len(chars))
            char_indices = dict((c, i) for i, c in enumerate(chars))
            indices_char = dict((i, c) for i, c in enumerate(chars))

            # cut the text in semi-redundant sequences of maxlen characters
            step = 3
            sentences = []
            next_chars = []
            for i in range(0, len(text) - self.maxlen, step):
                sentences.append(text[i: i + self.maxlen])
                next_chars.append(text[i + self.maxlen])
            print('nb sequences:', len(sentences))

            print('Vectorization...')
            x = np.zeros((len(sentences), self.maxlen, len(chars)), dtype=np.bool)
            y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
            for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):
                    x[i, t, char_indices[char]] = 1
                y[i, char_indices[next_chars[i]]] = 1

            self.chars = chars
            self.char_indices = char_indices
            self.indices_char = indices_char

            self.oh_sentences = x
            self.oh_next_chars = y

    def build_model(self):
        # build the model: a single LSTM
        print('Build model...')
        model = Sequential()

        model.add(Embedding(1000, 64, input_length=10))
        model.add(LSTM(128, input_shape=(self.maxlen, len(self.chars))))
        model.add(Dense(len(self.chars), activation='softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        self.model = model

    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def on_epoch_end(self, epoch, _):
        with tf.device(D_CPU):
            # Function invoked at end of each epoch. Prints generated text.
            print()
            print('----- Generating text after Epoch: %d' % epoch)

            next_char = 'A'

            sys.stdout.write(next_char)

            for i in range(400):
                x_pred = np.zeros((1, self.maxlen, len(self.chars)))
                x_pred[0, 0, self.char_indices[next_char]] = 1

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, 0.8)
                next_char = self.indices_char[next_index]

                sys.stdout.write(next_char)
                sys.stdout.flush()

            print()

    def train(self):
        with tf.device(D_GPU):
            self.build_model()
            print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)

            self.model.fit(self.oh_sentences,
                           self.oh_next_chars,
                           batch_size=128,
                           epochs=60,
                           callbacks=[print_callback])

gen = CharGenerator()
gen.train()
