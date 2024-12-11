import torch
import torch.nn as nn

import math
import numpy as np
import random

def file_to_dataset(filename):
    dataset = []

    fi = open(filename, "r")
    for line in fi:
        parts = line.strip().split("\t")
        input_sentence = parts[0]
        output_sentence = parts[1]

        dataset.append([input_sentence, output_sentence])

    return dataset

def lr_positions_from_brackets(sentence):
    lr_positions = []
    words = sentence.split()

    roles = []
    path = []
    for word in words:
        if word == "[":
            path.append("L")
        elif word == "]":
            if len(path) > 1:
                path = path[:-2]
                path.append("R")
            else:
                if path == ["R"]:
                    path = ["DONE"]
                else:
                    path = ["OVERDONE"]
        else:
            roles.append("".join(path))
            if path[-1] == "L":
                path = path[:-1]
                path.append("R")

    assert path == ["DONE"]

    return roles


def remove_excess_spaces(sentence):
    if "  " in sentence:
        return remove_excess_spaces(sentence.replace("  ", " "))
    else:
        return sentence


def truncate_at_eos(sentence):
    words = sentence.split()
    new_words = []

    for word in words:
        if word == "EOS":
            break
        elif word == "SOS":
            continue
        else:
            new_words.append(word)

    return " ".join(new_words)

def bracketed_sentence_to_sentence(bracketed_sentence):
    sentence = bracketed_sentence.replace("[", "").replace("]", "").strip()
    sentence = remove_excess_spaces(sentence)

    return sentence

def standardize_sentence(sentence):
    sentence = bracketed_sentence_to_sentence(sentence)
    sentence = truncate_at_eos(sentence)
    return sentence




