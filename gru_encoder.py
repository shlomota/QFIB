import torch
import numpy as np
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab, device="cpu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.device = device

        self.embedding = nn.Embedding(len(self.vocab.w2i), input_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=2)

    def forward(self, input):
        ### YOUR CODE HERE
        embedded = self.embedding(input)
        output, h = self.gru(embedded)
        ### --------------
        return output, h
