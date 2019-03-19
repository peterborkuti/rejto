# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

# Load Larger LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.callbacks import LambdaCallback

import tensorflow as tf

D_GPU = '/gpu:0'
D_CPU = '/cpu:0'


# load unidecoded ascii text and prepare
class Data:
    X = None
    y = None
    char_to_int = None
    int_to_char = None
    dataX = None
    n_vocab = None

    def __init__(self,
                 path='../books/unidecoded_rejto.txt',
                 text_size=20000, testString=None):
        with tf.device(D_CPU):
            raw_text = testString
            if raw_text is None:
                raw_text = open(path).read(text_size)

            # create mapping of unique chars to integers, and a reverse mapping
            chars = sorted(list(set(raw_text)))
            char_to_int = dict((c, i) for i, c in enumerate(chars))
            int_to_char = dict((i, c) for i, c in enumerate(chars))
            # summarize the loaded data
            n_chars = len(raw_text)
            n_vocab = len(chars)
            print("Total Characters: ", n_chars)
            print("Total Vocab: ", n_vocab)
            # prepare the dataset of input to output pairs encoded as integers
            seq_length = 100
            dataX = []
            dataY = []
            for i in range(0, n_chars - seq_length, 1):
                seq_in = raw_text[i:i + seq_length]
                seq_out = raw_text[i + seq_length]
                dataX.append([char_to_int[char] for char in seq_in])
                dataY.append(char_to_int[seq_out])
            n_patterns = len(dataX)
            print("Total Patterns: ", n_patterns)
            # reshape X to be [samples, time steps, features]
            X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
            # normalize
            X = X / float(n_vocab)
            # one hot encode the output variable
            y = np_utils.to_categorical(dataY)

            self.X = X
            self.y = y
            self.char_to_int = char_to_int
            self.int_to_char = int_to_char
            self.dataX = dataX
            self.n_vocab = n_vocab

        def get_random_int_data(self):
            start = numpy.random.randint(0, len(self.dataX)-1)

            return self.dataX[start]

        def str_to_intarr(self, data):
            return [self.char_to_int[char] for char in data]

        def ints_to_str(self, data):
            return ''.join([self.int_to_char[value] for value in data])


# define the LSTM model
class RNN:
    model = None

    def __init__(self, XShape, yShape):
        model = Sequential()
        model.add(
            LSTM(256,
                 input_shape=(XShape[1], XShape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(256))
        model.add(Dropout(0.2))
        model.add(Dense(yShape[1], activation='softmax'))

        self.model = model
        self.compile()

    @staticmethod
    def getFilename(filename: str=None):
        if (filename is None):
            filename = "model_wights"

        return filename + ".hdf5"

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def load_wights(self, filename):
        fileName = self.getFilename(filename)
        self.model.load_weights(fileName)
        self.compile()

    def save_wights(self, filename=None, loss=None):
        if loss is None:
            loss = ""

        filename = self.getFilename()

        self.model.save_wights(filename)

    def generate(self, data: Data, seed: str=None, seq_length=1000):
        if seed is None:
            pattern = data.get_random_int_data()
            seed = data.ints_to_str(pattern)
        else:
            pattern = data.str_to_intarr(seed)

        print("Seed:")
        print("\"", seed, "\"")
        result = []
        # generate characters
        for i in range(seq_length):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(data.n_vocab)
            prediction = self.model.predict(x, verbose=0)
            index = numpy.argmax(prediction)

            result.append(data.int_to_char[index])

            pattern.append(index)
            pattern = pattern[1:]
        print("\nDone.")

        return ''.join(result)


class Trainer:
    data = None
    rnn = None

    def __init__(self):
        self.data = Data()
        self.rnn = RNN(self.data.X.shape, self.data.y.shape)

    def on_epoch_end(self):
        print(self.rnn.generate(self.data))

    def train(self):
        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint_cb = ModelCheckpoint(filepath,
                                        monitor='loss',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='min')
        print_cb = LambdaCallback(on_epoch_end=self.on_epoch_end)
        callbacks_list = [checkpoint_cb, print_cb]

        self.rnn.model.fit(self.data.X,
                           self.data.y, epochs=20, batch_size=128,
                           callbacks=callbacks_list)

trainer = Trainer()
trainer.train()
