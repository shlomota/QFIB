import torch
import numpy as np
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, vocab, device="cpu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.device = device

        self.embedding = nn.Embedding(len(self.vocab.w2i), embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True, num_layers=2, bidirectional=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(2 * hidden_size, output_size)

    def forward(self, input):
        ### YOUR CODE HERE
        embedded = self.embedding(input)
        output, h = self.gru(embedded)
        # output = self.linear(self.relu(output))
        output = self.linear(output)
        ### --------------
        return output, h
