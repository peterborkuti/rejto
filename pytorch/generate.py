# https://github.com/spro/practical-pytorch

import unidecode
import string
import random
import time
import math
import torch
from torch.autograd import Variable
import torch.nn as nn

import os
import os.path


# Reading and un-unicode-encoding data

class Util:
    all_characters = string.printable
    n_characters = len(all_characters)

    @staticmethod
    def getDevice(verbose = True):
        device = "cpu"
        deviceName = "cpu"
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            deviceName = torch.cuda.get_device_name(device)
        
        if verbose:
            print ("Device:", device, deviceName)

        return device, deviceName

    @staticmethod
    def read_file(filename):
        file = unidecode.unidecode(open(filename).read())
        
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

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=3):
        super(RNN, self).__init__()
        self.device, _ = Util.getDevice()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size).to(self.device)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers).to(self.device)
        self.decoder = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, input, hidden):
        input = input.to(self.device)
        input = self.encoder(input.view(1, -1)).to(self.device)
        hidden = hidden.to(self.device)
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        hidden = hidden.to(self.device)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

    def generate(self, prime_str='A', predict_len=100, temperature=0.8):
        hidden = self.init_hidden()
        prime_input = Util.char_tensor(prime_str).to(self.device)
        predicted = prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str) - 1):
            _, hidden = self.decoder(prime_input[p], hidden)
        
        hidden = hidden.to(self.device)
        inp = prime_input[-1]
        
        for p in range(predict_len):
            output, hidden = self.forward(inp, hidden)
            
            # Sample from the network as a multinomial distribution

            # output_dist goes to inf
            # output_dist = output.data.view(-1).div(temperature).exp()
            data = output.data.view(-1).div(temperature)
            output_dist = torch.exp(data - data.max())

            top_i = torch.multinomial(output_dist, 1)[0]
            
            # Add predicted character to string and use as next input
            predicted_char = Util.all_characters[top_i]
            predicted += predicted_char
            inp = Util.char_tensor(predicted_char)

        return predicted

# https://github.com/spro/practical-pytorch



class Train:
    n_epochs = 2000
    print_every = 10
    hidden_size = 100
    n_layers = 3
    learning_rate = 0.03
    chunk_len = 1500

    start = time.time()
    all_losses = []
    loss_avg = 0

    def __init__(self, path=None, filename=None, file=None, loadPt=False):
        self.device, deviceName = Util.getDevice()

        if deviceName == "Tesla K80":
            print("Running on cloud " + deviceName)
            self.path = "/content/gdrive/My Drive/rejto/"
        else:
            print("Running on " + deviceName)
            self.path = "./"

        if filename is None:
            self.filename = "rejto.txt"
        
        if filename is not None:
            self.filename = filename
        
        self.filename = self.path + self.filename
        
        if file is None:
            self.file, self.file_len = Util.read_file(self.filename)
        else:
            self.file = file
            self.file_len = len(file)
        
        self.file = Util.char_tensor(self.file)

        isSavedDecoder = os.path.exists(self.getFileNameBase()+ '.pt')

        if loadPt and isSavedDecoder:
            self.decoder = torch.load(self.path + self.getFileNameBase() + '.pt').to(self.device)
        else:
            self.decoder = RNN(Util.n_characters, self.hidden_size, Util.n_characters, self.n_layers).to(self.device)

        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def getFileNameBase(self):
        return os.path.splitext(os.path.basename(self.filename))[0]

    def random_training_set(self, file, chunk_len):
        start_index = random.randint(0, self.file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = torch.tensor(file[start_index:end_index], dtype=torch.long)

        return Variable(chunk)

    def train(self, inputToLearn):
        if isinstance(inputToLearn, str):
            inputToLearn = Util.char_tensor(inputToLearn)
        
        inp = inputToLearn[:-1]
        target = inputToLearn[1:]

        chunk_len = len(inp)
        loss = 0

        hidden = self.decoder.init_hidden().to(self.device)
        self.decoder.zero_grad()

        for c in range(chunk_len):
            output, hidden = self.decoder(inp[c], hidden)
            output = output.to(self.device)
            target = target.to(self.device)
            loss += self.criterion(output, target[c].view(1))

        loss.backward()
        self.decoder_optimizer.step()

        return loss.data.item() / chunk_len

    def save(self, loss):
        lss = str(loss).replace('.', '_')
        save_filename = self.getFileNameBase() + '-' + lss +'.pt'
        torch.save(self.decoder, self.path + save_filename)
        print('Saved as %s' % save_filename)

    def start_training(self):
        loss_avg = 0
        loss = 1000
        try:
            print("Training for %d epochs..." % self.n_epochs)
            for epoch in range(1, self.n_epochs + 1):
                loss = self.train(*self.random_training_set(self.file, self.chunk_len))
                loss_avg += loss

                if epoch % self.print_every == 0:
                    print('[%s (%d %d%%) %.4f]' % (Util.time_since(self.start), epoch, epoch / self.n_epochs * 100, loss))
                    print(self.decoder.generate('A', 100), '\n')

            print("Saving...")
            self.save(loss)
        except KeyboardInterrupt:
            print("Saving before quit...")
            self.save('')
