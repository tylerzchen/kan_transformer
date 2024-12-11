import torch
import torch.nn as nn

import math
import numpy as np
import random
from transformersource import TransformerEncoder, Transformer, TransformerDecoder, TransformerDecoderLayer, TransformerEncoderLayer

from utils import *

word2index = {}
index2word = {}

fi = open("vocab.txt", "r")
for index, line in enumerate(fi):
    word = line.strip()
    word2index[word] = index
    index2word[index] = word
fi.close()

# From official PyTorch implementation
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, sentence):
        return self.pe[:x.size(0), :]

def lr_position_to_encoding(lr_position, reps=10, dim=24):
    final_emb = []
    for i in range(reps):
        count = 0
        for lr in lr_position:
            if lr == 'L':
                if(count < dim):
                    final_emb.append(1)
                    count += 1
                if(count < dim):
                    final_emb.append(0)
                    count +=1
            elif lr == 'R':
                if(count < dim):
                    final_emb.append(0)
                    count += 1
                if(count < dim):
                    final_emb.append(1)
                    count += 1
        if(count < dim):
            for j in range(dim - count):
                final_emb.append(0)
    return final_emb



class TreePositionalEncoding(nn.Module):
    def __init__(self, hidden_size):
        super(TreePositionalEncoding, self).__init__()

        self.hidden_size = hidden_size

    def forward(self, word_emb, bracketed_sentence):
        lr_positions = lr_positions_from_brackets(bracketed_sentence)

        dim = self.hidden_size // 10
        reps = 10

        pos_encoding = []

        for lr_position in lr_positions:
            lr_seq = list(lr_position)

            emb = lr_position_to_encoding(lr_seq, reps=reps, dim=dim)

            pos_encoding.append([emb])

        return torch.FloatTensor(pos_encoding)

class EncoderTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, tree=False):
        super(EncoderTransformer, self).__init__()

        self.n_layers = 4
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_head = 4
        self.dim_feedforward = 4*self.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.tree = tree
        if tree:
            self.positional_encoder = TreePositionalEncoding(self.hidden_size)
        else:
            self.positional_encoder = PositionalEncoding(self.hidden_size)

        encoder_layer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_head, dim_feedforward=self.dim_feedforward, dropout=0)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.n_layers)

    def forward(self, sentence):

        if self.tree:
            words = standardize_sentence(sentence).split()
        else:
            words = sentence.split()

        input_seq = torch.LongTensor([[word2index[word] for word in words]]).transpose(0,1)
        emb = self.embedding(input_seq)

        emb = emb + self.positional_encoder(emb, sentence)

        memory = self.transformer_encoder(emb)

        return memory



class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_length=30, tree=False):
        super(DecoderTransformer, self).__init__()

        self.n_layers = 4
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_head = 4
        self.dim_feedforward = 4*self.hidden_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.tree = tree
        if tree:
            self.positional_encoder = TreePositionalEncoding(self.hidden_size)
        else:
            self.positional_encoder = PositionalEncoding(self.hidden_size)

        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_size, nhead=self.n_head, dim_feedforward=self.dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=self.n_layers)

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, memory, target_output=None):

        if target_output is not None:
            if self.tree:
                input_words = ["SOS"] + standardize_sentence(target_output).split()
            else:
                input_words = ["SOS"] + target_output.split() 
            input_tensor = torch.tensor([[word2index[word] for word in input_words]]).transpose(0,1)

            emb = self.embedding(input_tensor)
            positional_emb = self.positional_encoder(emb, target_output)
            if self.tree:
                positional_emb = torch.cat([torch.zeros_like(positional_emb[0]).unsqueeze(0), positional_emb], dim=0)
            emb = emb + positional_emb

            tgt_mask = Transformer.generate_square_subsequent_mask(emb.shape[0])
            output = self.transformer_decoder(emb, memory, tgt_mask=tgt_mask)
            output = self.out(output).squeeze(1)

            words = []
            output_vectors = []
            for output_vector in output:
                topv, topi = torch.topk(output_vector, 1)
                word = index2word[topi.item()]
                words.append(word)
                output_vectors.append(output_vector.unsqueeze(0))

        else:
            input_words = ["SOS"]
            output_vectors = []
            done = False
            while not done:
                input_tensor = torch.tensor([[word2index[word] for word in input_words]]).transpose(0,1)

                emb = self.embedding(input_tensor)
                emb = emb + self.positional_encoder(emb, None)

                tgt_mask = Transformer.generate_square_subsequent_mask(emb.shape[0])
                output = self.transformer_decoder(emb, memory, tgt_mask=tgt_mask)
                output = self.out(output).squeeze(1)

                final_output = output[-1]
                topv, topi = torch.topk(final_output, 1)
                word = index2word[topi.item()]
                input_words.append(word)
                output_vectors.append(final_output.unsqueeze(0))

                if word == "EOS" or len(input_words) > self.max_length:
                    done = True

            words = input_words[1:]

        return output_vectors, " ".join(words)



class Seq2SeqTransformer(nn.Module):

    def __init__(self, vocab_size, hidden_size, max_length=30, tree=False):
        super(Seq2SeqTransformer, self).__init__()

        self.encoder = EncoderTransformer(vocab_size, hidden_size, tree=tree)
        self.decoder = DecoderTransformer(vocab_size, hidden_size, max_length=max_length, tree=tree)


    def forward(self, input_sentence, target_sentence=None):

        encoding = self.encoder(input_sentence)
        output_vectors, output_sentence = self.decoder(encoding, target_output=target_sentence)

        return output_vectors, output_sentence



if __name__ == "__main__":
    print("L", lr_position_to_encoding("L", reps=2, dim=6))
    print("R", lr_position_to_encoding("R", reps=2, dim=6))
    print("LLR", lr_position_to_encoding("LLR", reps=2, dim=6))
    print("RRL", lr_position_to_encoding("RRL", reps=2, dim=6))
    print("RRLL", lr_position_to_encoding("RRLL", reps=2, dim=6))
    print("R", lr_position_to_encoding("R", reps=4, dim=4))