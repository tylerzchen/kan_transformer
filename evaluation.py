

import torch
import torch.nn as nn

import math
import numpy as np
import random

from srn import Seq2SeqSRN
from gru import Seq2SeqGRU
from lstm import Seq2SeqLSTM
from transformer import Seq2SeqTransformer
from utils import standardize_sentence, file_to_dataset


word2index = {}
index2word = {}

fi = open("vocab.txt", "r")
for index, line in enumerate(fi):
    word = line.strip()
    word2index[word] = index
    index2word[index] = word
fi.close()

def compute_loss(output_vectors, target_sentence):
    loss_function = nn.CrossEntropyLoss()

    target_words = standardize_sentence(target_sentence).split() + ["EOS"]

    total_loss = 0
    count_losses = 0
    for output_vector, target_word in zip(output_vectors, target_words):
        target_index = word2index[target_word]

        # Compute loss as a tensor
        loss = loss_function(output_vector, torch.tensor([target_index], dtype=torch.long))

        total_loss += loss
        count_losses += 1

    # Return the mean loss as a tensor, not a scalar
    return total_loss / count_losses


def compute_loss_on_dataset(model, dataset, provide_target=True):
    total_loss = 0
    total_correct = 0
    count_sentences = 0

    for example in dataset:
        input_sentence = example[0]
        target_sentence = example[1]

        if provide_target:
            # Ensure the model returns tensors
            output_vectors, output_sentence = model(input_sentence, target_sentence=target_sentence)
        else:
            output_vectors, output_sentence = model(input_sentence)

        # Compute loss and ensure it is a tensor
        loss = compute_loss(output_vectors, target_sentence)
        if isinstance(loss, torch.Tensor):
            total_loss += loss.detach()
        else:
            print("DEBUG: Loss is not a tensor!", type(loss))


        # Compute accuracy
        if standardize_sentence(output_sentence) == standardize_sentence(target_sentence):
            total_correct += 1

        count_sentences += 1

    # Return average loss and accuracy
    average_loss = total_loss / count_sentences
    accuracy = total_correct / count_sentences

    return average_loss, accuracy


def print_n_examples(model, dataset, n, provide_target=True):

    # Prints n examples of the output predicted by the model, printed
    # next to the correct output
    print("EXAMPLE MODEL OUTPUTS")
    for example in dataset[:n]:
        input_sentence = example[0]
        target_sentence = example[1]

        if provide_target:
            output_vectors, output_sentence = model(input_sentence, target_sentence=target_sentence)
        else:
            output_vectors, output_sentence = model(input_sentence)

        print("CORRECT:", standardize_sentence(target_sentence))
        print("OUTPUT: ", standardize_sentence(output_sentence))
        print("")

def first_word_accuracy(model, dataset, provide_target=True):

    total_correct = 0
    count_examples = 0

    for example in dataset:
        input_sentence = example[0]
        target_sentence = example[1]

        if provide_target:
            output_vectors, output_sentence = model(input_sentence, target_sentence=target_sentence)
        else:
            output_vectors, output_sentence = model(input_sentence)

        correct_first_word = standardize_sentence(target_sentence).split()[0]
        predicted_first_word = standardize_sentence(output_sentence).split()[0]

        if predicted_first_word == correct_first_word:
            total_correct += 1
        count_examples += 1

    return total_correct / count_examples



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset (e.g., test or gen)", type=str, default="test")
    parser.add_argument("--hidden_size", help="size of the hidden layer", type=int, default=240)
    parser.add_argument("--model_name", help="name for your model", type=str, default=None)
    parser.add_argument("--architecture", help="model type", type=str, default=None)
    args = parser.parse_args()

    if args.architecture == "SRN":
        model = Seq2SeqSRN(len(word2index), args.hidden_size)
        dataset = file_to_dataset("data/question." + args.dataset)
    elif args.architecture == "GRU":
        model = Seq2SeqGRU(len(word2index), args.hidden_size)
        dataset = file_to_dataset("data/question." + args.dataset)
    elif args.architecture == "LSTM":
        model = Seq2SeqLSTM(len(word2index), args.hidden_size)
        dataset = file_to_dataset("data/question." + args.dataset)
    elif args.architecture == "Transformer":
        model = Seq2SeqTransformer(len(word2index), args.hidden_size)
        dataset = file_to_dataset("data/question." + args.dataset)
    elif args.architecture == "Tree":
        model = Seq2SeqTransformer(len(word2index), args.hidden_size, tree=True)
        dataset = file_to_dataset("data/question_bracket." + args.dataset)
    elif args.architecture == "TreeEmpty":
        model = Seq2SeqTransformer(len(word2index), args.hidden_size, tree=True)
        dataset = file_to_dataset("data/question_bracket_empty." + args.dataset)

    model.load_state_dict(torch.load("weights/" + args.model_name + ".weights"))

    if args.architecture == "Tree" or args.architecture == "TreeEmpty":
        provide_target = True
    else:
        provide_target = False

    print_n_examples(model, dataset, 10)

    loss, accuracy = compute_loss_on_dataset(model, dataset, provide_target=provide_target)
    print("FULL-SENTENCE ACCURACY:", accuracy)

    first_word_acc = first_word_accuracy(model, dataset, provide_target=provide_target)
    print("FIRST-WORD ACCURACY", first_word_acc)


