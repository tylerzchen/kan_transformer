

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

class MySRNUnit(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size):
        super(MySRNUnit, self).__init__()

        # Weight matrix applied to the word embedding
        self.wx = nn.Linear(emb_size, hidden_size)

        # Weight matrix applied to the previous hidden state
        self.wh = nn.Linear(hidden_size, hidden_size)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, word_emb, hidden):

        hidden = self.wx(word_emb) + self.wh(hidden)
        hidden = self.sigmoid(hidden)

        return hidden



class EncoderSRN(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size):
        super(EncoderSRN, self).__init__()

        self.hidden_size = hidden_size

        # Emedding layer: Takes in a word's numerical ID and returns an embedding for it
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # Recurrent unit (the part that updates the hidden state)
        self.srn = MySRNUnit(vocab_size, emb_size, hidden_size)

    def init_hidden(self):

        # Function for creating the initial hidden state, which is all zeroes
        # in the length of self.hidden_size
        hidden = torch.zeros(self.hidden_size)

        return hidden

    def forward(self, sentence):

        words = sentence.split()

        # Initialize the hidden state
        hidden = self.init_hidden()

        # Loop over the input words
        for word in words:

            # Embed the word
            word_emb = self.embedding(torch.tensor([word2index[word]]))

            # Update the hidden state based on the new word embedding
            # and the previous hidden state.
            # Note: No need to include a bias term here, because nn.Linear()
            # (which is what self.wx and self.wh are) automatically includes
            # a bias term
            hidden = self.srn(word_emb, hidden) 


        # Return the final hidden state
        return hidden


class DecoderSRN(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, max_length=30):
        super(DecoderSRN, self).__init__()

        self.hidden_size = hidden_size

        # Emedding layer: Takes in a word's numerical ID and returns an embedding for it
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # Weight matrix applied to the word embedding
        self.wx = nn.Linear(emb_size, hidden_size)

        # Weight matrix applied to the previous hidden state
        self.wh = nn.Linear(hidden_size, hidden_size)

        # Output layer: Goes from hidden state to predicted output
        self.out_layer = nn.Linear(hidden_size, vocab_size)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

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
                # Embed the previous output
                word_emb = self.embedding(torch.tensor([word2index[previous_word]]))

                # Update the hidden state
                hidden = self.sigmoid(self.wx(word_emb) + self.wh(hidden))

                # Produce the output - a vector that is the same size as the
                # vocabulary. We don't need to include a softmax here 
                # because the loss function incorporates the softmax
                output_vector = self.out_layer(hidden)
                
                output_vectors.append(output_vector)

                # Figure out what word has the highest probability, and use
                # that as the output word for this timestep
                topv, topi = torch.topk(output_vector, 1)
                output_word = index2word[topi.item()]
                output_words.append(output_word)
                
                if output_word == "EOS" or len(output_words) > self.max_length:
                    done = True

                previous_word = output_word
        
        else:
            # Similar as above, but now we use the target output at timestep t-1
            # (rather than what the model produced) as input at timestep t
            target_words = target_output.split() + ["EOS"]
            for target_word in target_words:
                word_emb = self.embedding(torch.tensor([word2index[previous_word]]))
                hidden = self.sigmoid(self.wx(word_emb) + self.wh(hidden))
                output_vector = self.out_layer(hidden)

                output_vectors.append(output_vector)

                topv, topi = torch.topk(output_vector, 1)
                output_word = index2word[topi.item()]
                output_words.append(output_word)

                previous_word = target_word


        output_sentence = " ".join(output_words)

        return output_vectors, output_sentence


# A class that puts together the encoder and decoder defined above
class Seq2SeqSRN(nn.Module):

    def __init__(self, vocab_size, hidden_size, max_length=30):
        super(Seq2SeqSRN, self).__init__()

        self.encoder = EncoderSRN(vocab_size, hidden_size, hidden_size)
        self.decoder = DecoderSRN(vocab_size, hidden_size, hidden_size, max_length=max_length)


    def forward(self, input_sentence, target_sentence=None):

        encoding = self.encoder(input_sentence)
        output_vectors, output_sentence = self.decoder(encoding, target_output=target_sentence)

        return output_vectors, output_sentence




