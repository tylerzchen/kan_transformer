

import torch
import torch.nn as nn

import math
import numpy as np
import random

from srn import Seq2SeqSRN
from gru import Seq2SeqGRU
from lstm import Seq2SeqLSTM
from transformer import Seq2SeqTransformer
from utils import file_to_dataset
from evaluation import compute_loss_on_dataset, compute_loss


word2index = {}
index2word = {}

fi = open("vocab.txt", "r")
for index, line in enumerate(fi):
    word = line.strip()
    word2index[word] = index
    index2word[index] = word
fi.close()



# Train a model on a training set, and save its weights
def train_model(model, training_set, validation_set, model_name, lr=0.00005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = math.inf
    for index, example in enumerate(training_set):
        if index % 1000 == 0:
            loss, accuracy = compute_loss_on_dataset(model, validation_set)
            print(index, "Acc:", accuracy, "Loss:", loss)

            if loss < best_loss:
                torch.save(model.state_dict(), model_name + ".weights")
                best_loss = loss


        input_sentence = example[0]
        target_sentence = example[1]

        output_vectors, output_sentence = model(input_sentence, target_sentence=target_sentence)

        loss = compute_loss(output_vectors, target_sentence)
        assert isinstance(loss, torch.Tensor), "Loss must be a tensor for backward()"
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", help="size of the hidden layer", type=int, default=240)
    parser.add_argument("--model_name", help="name for your model", type=str, default=None)
    parser.add_argument("--architecture", help="model type: SRN, LSTM, or Transformer", type=str, default=None)
    args = parser.parse_args()


    # Create our architecture, then select an appropriate learning
    # weight and the appropriate dataset
    if args.architecture == "SRN":
        model = Seq2SeqSRN(len(word2index), args.hidden_size)
        lr = 0.001

        training_set = file_to_dataset("data/question.train")
        validation_set = file_to_dataset("data/question.dev")

    elif args.architecture == "GRU":
        model = Seq2SeqGRU(len(word2index), args.hidden_size)
        lr = 0.001
        
        training_set = file_to_dataset("data/question.train")
        validation_set = file_to_dataset("data/question.dev")

    elif args.architecture == "LSTM":
        model = Seq2SeqLSTM(len(word2index), args.hidden_size)
        lr = 0.001
        
        training_set = file_to_dataset("data/question.train")
        validation_set = file_to_dataset("data/question.dev")

    elif args.architecture == "Transformer":
        model = Seq2SeqTransformer(len(word2index), args.hidden_size)
        lr = 0.00005

        training_set = file_to_dataset("data/question.train")
        validation_set = file_to_dataset("data/question.dev")

    elif args.architecture == "Tree":
        model = Seq2SeqTransformer(len(word2index), args.hidden_size, tree=True)
        lr = 0.0001

        training_set = file_to_dataset("data/question_bracket.train")
        validation_set = file_to_dataset("data/question_bracket.dev")

    elif args.architecture == "TreeEmpty":
        model = Seq2SeqTransformer(len(word2index), args.hidden_size, tree=True)
        lr = 0.0001

        training_set = file_to_dataset("data/question_bracket_empty.train")
        validation_set = file_to_dataset("data/question_bracket_empty.dev")

    train_model(model, training_set, validation_set, "weights/" + args.model_name, lr=lr)




