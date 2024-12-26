# Kolmogorov-Arnold Neural Network (KAN) Transformer

This repository contains the implementation of the **Kolmogorov-Arnold Neural Network (KAN) Transformer**, a novel transformer architecture that leverages the Kolmogorov-Arnold Representation Theorem. It introduces rational and Taylor polynomial base functions in the feed-forward layers to achieve a balance of memory efficiency and performance. The repository benchmarks the KAN transformer against standard transformers, LSTMs, and tree transformers on sequence-to-sequence (seq2seq) tasks.

## Overview

The Kolmogorov-Arnold Representation Theorem states that any multivariate continuous function can be expressed as a sum of single-variable functions. Building on this theorem, the KAN transformer replaces traditional feed-forward layers with layers inspired by the theorem. Two variations are explored:
- **Rational Base Functions**
- **Taylor Polynomial Base Functions**

These modifications aim to provide theoretical expressiveness while maintaining computational efficiency.

### Key Features

- Implementation of the KAN transformer using PyTorch.
- Two base functions for feed-forward layers: Rational and Taylor Polynomial.
- Benchmarked on syntactic transformation tasks, such as English question formation.
- Comparative analysis with LSTMs, standard transformers, and tree transformers.

## Benchmarked Tasks

The primary task is **English question formation**. For example:
- Input: *The walrus doesn’t read.*
- Expected Output: *Doesn’t the walrus read?*

The benchmarks include two datasets:
1. **Test Set**: Syntactically similar examples to the training set.
2. **Generalization Set**: Syntactically different examples requiring hierarchical rule-based transformations.

## Results

### Test Set Performance

| Architecture                     | First-Word Accuracy | Full Accuracy |
|----------------------------------|---------------------|---------------|
| LSTM                             | 0.996               | 0.733         |
| Transformer                      | 0.999               | 0.998         |
| Tree Transformer (Without EMPTY) | 1.000               | 0.993         |
| Tree Transformer (With EMPTY)    | 1.000               | 0.999         |
| KAN with Rational Base Function  | 0.998               | 0.972         |
| KAN with Polynomial Base Function | n/a                 | n/a           |

### Generalization Set Performance

| Architecture                     | First-Word Accuracy | Full Accuracy |
|----------------------------------|---------------------|---------------|
| LSTM                             | 0.026               | 0.000         |
| Transformer                      | 0.060               | 0.000         |
| Tree Transformer (Without EMPTY) | 1.000               | 0.156         |
| Tree Transformer (With EMPTY)    | 1.000               | 0.847         |
| KAN with Rational Base Function  | 0.000               | 0.019         |
| KAN with Polynomial Base Function | n/a                 | n/a           |

### Observations

- The rational base function KAN performs well on the test set but struggles with generalization due to overfitting.
- The polynomial base function encountered exploding gradient issues, preventing successful training.

## Installation

Clone the repository:

```bash
git clone https://github.com/tylerzchen/kan_transformer.git
cd kan_transformer
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage:

# Training:
To train the KAN transformer:
```bash
python train.py --model_type kan --base_function rational
```

# Evaluation:
To evaluate a train model using the evaluation.py script ensure that the trained model weights are saved in the weights/ directory with the name <model_name>.weights. Then, run the following command:
```bash
python evaluation.py --dataset <test|gen> --hidden_size <hidden_size> --model_name <model_name> --architecture <architecture>
```
Where:
Arguments
- --dataset: Specify the dataset to evaluate on. Options are test or gen.
- --hidden_size: Hidden layer size used during training. Default is 240.
- --model_name: Name of the saved model weights file (without the .weights extension).
- --architecture: Specify the model architecture. Options include:
  * SRN: Simple Recurrent Network
  * GRU: Gated Recurrent Unit
  *  LSTM: Long Short-Term Memory
  *  Transformer: Standard Transformer
  *  Tree: Tree Transformer (without EMPTY nodes)
  *  TreeEmpty: Tree Transformer (with EMPTY nodes)

## Citation:
If you use this work please cite:
Vincent Li, Jared Wyetzner, Tyler Chen, "A Benchmarking of Kolmorogorov-Arnold Transformers with Taylor Polynomial Base Function on Seq2Seq Syntactic Transformation Tasks"
