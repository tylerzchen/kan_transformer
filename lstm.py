

import torch
import torch.nn as nn

import math
import numpy as np
import random


word2index = {}
index2word = {}

fi = open("vocab.txt", "r")
for index, line in enumerate(fi):
    word = line.strip()
    word2index[word] = index
    index2word[index] = word
fi.close()

class MyLSTMUnit(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size):
        super(MyLSTMUnit, self).__init__()
        self.wfx = nn.Linear(emb_size, hidden_size)
        self.wix = nn.Linear(emb_size, hidden_size)
        self.wgx = nn.Linear(emb_size, hidden_size)
        self.wox = nn.Linear(emb_size, hidden_size)

        self.wfh = nn.Linear(hidden_size, hidden_size)
        self.wih = nn.Linear(hidden_size, hidden_size)
        self.wgh = nn.Linear(hidden_size, hidden_size)
        self.woh = nn.Linear(hidden_size, hidden_size)
        # defining sigmoids and tanh
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, word_emb, hidden):
        ht, ct = hidden
        ft = self.sigmoid(self.wfx(word_emb) + self.wfh(ht))
        it = self.sigmoid(self.wix(word_emb) + self.wih(ht))
        gt = self.tanh(self.wgx(word_emb) + self.wgh(ht))
        ct = ft * ct + it * gt
        ot = self.sigmoid(self.wox(word_emb) + self.woh(ht))
        ht = ot * self.tanh(ct)


        hidden = [ht, ct]

        return hidden



class EncoderLSTM(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size):
        super(EncoderLSTM, self).__init__()

        self.hidden_size = hidden_size

        # Emedding layer: Takes in a word's numerical ID and returns an embedding for it
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # The recurrent unit that updates the hidden state
        self.lstm = MyLSTMUnit(vocab_size, emb_size, hidden_size)

    def init_hidden(self):

        # Function for creating the initial hidden state, which is all zeroes
        # in the length of self.hidden_size
        # Note that the LSTM's hidden state contains two separate vectors: h_t and c_t
        hidden = [torch.zeros(self.hidden_size), torch.zeros(self.hidden_size)]

        return hidden

    def forward(self, sentence):

        words = sentence.split()

        # Initialize the hidden state
        hidden = self.init_hidden()

        # Loop over the input words
        for word in words:

            # Embed the word
            word_emb = self.embedding(torch.tensor([word2index[word]]))

            # Update the hidden state by applying our LSTM unit
            # to the word embedding and the previous hidden state
            hidden = self.lstm(word_emb, hidden)


        # Return the final hidden state
        return hidden


class DecoderLSTM(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, max_length=30):
        super(DecoderLSTM, self).__init__()


        # Emedding layer: Takes in a word's numerical ID and returns an embedding for it
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # Recurrent unit that updates the hidden state
        self.lstm = MyLSTMUnit(vocab_size, emb_size, hidden_size)

        # Output layer: Goes from hidden state to predicted output
        self.out_layer = nn.Linear(hidden_size, vocab_size)

        self.max_length = max_length

    def forward(self, encoding, target_output=None):

        previous_word = "SOS"
        output_vectors = []
        output_words = []

        # Initialize the hidden state
        hidden = encoding

        # Two possibilities:
        # - A target output is provided. In this case, the new input that the
        #   model receives at each timestep is whatever would have been the
        #   correct output at the previous timestep (even if the model didn't
        #   get that output)
        # - A target output is not provided. In this case, the model's output at
        #   timestep t-1 is used as its input at timestep t.
        if target_output is None:
            done = False

            while not done:

                # Embed the previous word
                word_emb = self.embedding(torch.tensor([word2index[previous_word]]))

                # Update the hidden state
                hidden = self.lstm(word_emb, hidden)
               
                # Produce the output - a vector that is the same size as the
                # vocabulary. We don't need to include a softmax here 
                # because the loss function incorporates the softmax
                # Note that the output is predicted only based on the first sub-component
                # of the LSTM's hidden state
                output_vector = self.out_layer(hidden[0])
                output_vectors.append(output_vector)

                # Figure out what word has the highest probability, and use
                # that as the output word for this timestep (and the input word
                # for the next timestep)
                topv, topi = torch.topk(output_vector, 1)
                output_word = index2word[topi.item()]
                output_words.append(output_word)
                
                # Stop when we produce EOS, or when we hit the max length
                if output_word == "EOS" or len(output_words) > self.max_length:
                    done = True

                previous_word = output_word
        
        else:

            # Similar as above, but now we use the target output at timestep t-1
            # (rather than what the model produced) as input at timestep t

            target_words = target_output.split() + ["EOS"]
            for target_word in target_words:
                word_emb = self.embedding(torch.tensor([word2index[previous_word]]))
                hidden = self.lstm(word_emb, hidden)
                output_vector = self.out_layer(hidden[0])

                output_vectors.append(output_vector)

                topv, topi = torch.topk(output_vector, 1)
                output_word = index2word[topi.item()]
                output_words.append(output_word)

                previous_word = target_word


        output_sentence = " ".join(output_words)

        return output_vectors, output_sentence

# A class that puts together the encoder and decoder defined above
class Seq2SeqLSTM(nn.Module):

    def __init__(self, vocab_size, hidden_size, max_length=30):
        super(Seq2SeqLSTM, self).__init__()

        self.encoder = EncoderLSTM(vocab_size, hidden_size, hidden_size)
        self.decoder = DecoderLSTM(vocab_size, hidden_size, hidden_size, max_length=max_length)


    def forward(self, input_sentence, target_sentence=None):

        encoding = self.encoder(input_sentence)
        output_vectors, output_sentence = self.decoder(encoding, target_output=target_sentence)

        return output_vectors, output_sentence



