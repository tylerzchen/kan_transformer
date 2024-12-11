

import torch
from torch import nn
import math


# Complete this function by replacing each "None" 
# with functional code
def self_attention(word_embs, Wq, Wk, Wv, prnt=True):

    # First compute the query vectors. This should be a list of 
    # query vectors, where each vector is a PyTorch tensor. 
    # There should be one query vector for each word embedding
    # in the input.
    # For word embedding x, the query vector is the matrix Wq
    # multiplied by x.
    # Helpful hint: torch.mv(W, x) gives you a 
    # matrix-vector product between matrix W and vector x
    queries = [torch.mv(Wq, x) for x in word_embs]

    # Compute the key vectors, similarly to the query vectors
    keys = [torch.mv(Wk, x) for x in word_embs]
    
    # Compute the value vectors, similar to the key vectors
    values = [torch.mv(Wv, x) for x in word_embs]

    if prnt:
        print("Queries:")
        print(queries)
        print("")
        print("Keys:")
        print(keys)
        print("")
        print("Values:")
        print(values)
        print("")

    # Compute the context vectors: each context vector is
    # a weighted sum of the value vectors, where the weights
    # are determined using the queries and keys
    context_vectors = []
    for index, searching_word in enumerate(word_embs):
        query = queries[index]
        scores = []

        # FILL IN CODE THAT POPULATES "scores" WITH
        # THE ATTENTION SCORE BETWEEN "query" AND EACH
        # KEY IN "keys"
        # Helpful hint: In PyTorch, torch.dot(a,b) gives 
        # you the dot product between two vectors
        # We are using scaled dot-product attention as 
        # discussed in class: to get each score, you take the
        # dot product between the query and the key, then scale
        # it by dividing by the square root of the length of the
        # key vector, and finally you take a softmax over all 
        # the scores
        for key in keys:
            score = torch.dot(query, key)/math.sqrt(len(key))
            scores.append(score)
        scores = nn.Softmax(dim=0)(torch.tensor(scores))


        if prnt:
            print("Index", index, "scores:", scores)

        # FILL IN CODE THAT USES THE SCORES TO CREATE A
        # WEIGHTED SUM OF THE VALUE VECTORS.
        weighted_sum = sum(score.item() * value for score, value in zip(scores, values))

        context_vectors.append(weighted_sum)

    return context_vectors



Wq = torch.tensor([[-0.3, 0.5, -1.4, -1.4, 1.0],
        [0.0, 0.3, 0.5, -0.9, 0.4],
        [0.1, -0.4, 1.4, 1.2, -1.1]])

Wk = torch.tensor([[1.2, 0.4, -0.6, -0.6, 0.4],
        [0.1, 0.6, -0.5, 0.7, 1.2],
        [-0.5, 0.9, 1.3, 0.0, -1.1]])

Wv = torch.tensor([[-0.3, 2.4, 6.1, 6.2, 5.8],
        [-0.6, 4.3, 3.3,-5.7, 0.2],
        [2.3, -0.9, 3.6, -5.0, 3.5]])


word_emb_1 = [2.2, -1.4, 3.6, 0.5, -2.2]
word_emb_2 = [-1.9, 3.6, -0.4, 2.2, -3.8]
word_emb_3 = [0.4, 0.0, -2.3, 1.6, 0.5]
word_emb_4 = [0.6, -2.5, 3.6, -1.1, 2.8]

word_embs = [word_emb_1, word_emb_2, word_emb_3, word_emb_4]
word_embs = [torch.tensor(x) for x in word_embs]


context_vectors = self_attention(word_embs, Wq, Wk, Wv)

print("")
print("Context vectors:")
print(context_vectors)




