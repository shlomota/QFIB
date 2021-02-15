import torch
import numpy as np
import torch.nn as nn
import random


class DecoderSimple(nn.Module):
    def __init__(self, input_size, hidden_size, vocab, device="cpu", enc_hidden_size=256):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.output_size = len(self.vocab.w2i)
        self.uses_copying = False
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # for projecting the last hidden state of the encoder to the decoder space,
        # as the first decoder hidden state, in case the two dimensions don't match
        self.W_p = nn.Linear(enc_hidden_size, hidden_size)

        self.gru_cell = nn.GRUCell(self.input_size, self.hidden_size)

        # for output
        self.W_s = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, targets, h, evaluation_mode=False, **kwargs):
        ### YOUR CODE HERE
        embedded_targets = self.embedding(targets)

        outputs = list()
        predicted_outputs = list()

        first_input = self.embedding(torch.tensor([self.vocab.w2i["s"]]).to(self.device))
        current_h = self.W_p(h).squeeze(1)

        # almost always use teacher forcing if not evaluation, since evaluation uses exact match
        tf_thresh = 0.5


        # embedded_inputs = torch.cat([first_input.view(1, 1, -1), embedded_targets[:-1]])
        embedded_inputs = torch.cat([first_input.view(1, 1, -1), embedded_targets[:,:-1,:]], dim=1)
        # embedded_inputs = torch.cat([first_input, embedded_targets])
        if not evaluation_mode and random.random() > tf_thresh:
            # use teacher forcing
            for i in range(embedded_inputs.shape[1]):
            # for input in embedded_inputs[0, :, :]:
                input = embedded_inputs[:, i, :]
                current_h = self.gru_cell(input,current_h)
                current_output = self.W_s(current_h)
                outputs.append(current_output)
                predicted_outputs.append(torch.argmax(current_output))
        else:

            current_input = first_input
            # current_output_prediction = self.vocab.w2i["s"]
            for i in range(targets.shape[-1]):
                current_h = self.gru_cell(current_input,current_h)
                current_output = self.W_s(current_h)
                current_output_prediction = torch.argmax(current_output)
                predicted_outputs.append(current_output_prediction)
                current_input = self.embedding(current_output_prediction).view(1, -1)
                outputs.append(current_output)

        outputs = torch.cat(outputs)

        ### --------------
        return outputs