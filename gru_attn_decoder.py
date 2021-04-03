import torch
import numpy as np
import torch.nn as nn
import random
from gru_decoder_batch import DecoderSimple


class DecoderAttention(DecoderSimple):
    def __init__(self, input_size, hidden_size, vocab, device="cpu", enc_hidden_size=256):
        super().__init__(input_size, hidden_size, vocab, device)
        # for attention
        self.enc_hidden_size = enc_hidden_size
        self.W_a = nn.Linear(self.enc_hidden_size, self.hidden_size)

        # for output
        self.W_s = nn.Linear(self.hidden_size + self.enc_hidden_size, self.output_size)

    def forward(self, targets, h, evaluation_mode=False, **kwargs):
        if "enc_outputs" in kwargs:
            enc_outputs = kwargs["enc_outputs"]

        ### YOUR CODE HERE
        start_index = torch.tensor(self.vocab.w2i["s"]).to(self.device)
        outputs = list()


        batch_size = targets.shape[0]
        first_input = self.embedding(start_index)
        first_inputs = torch.stack([first_input] * batch_size)

        predicted_outputs = [torch.stack([start_index] * batch_size), ]
        current_h = self.W_p(h).squeeze(1)

        tf_thresh = 0.5
        attention_tensor = enc_outputs
        embedded_targets = self.embedding(targets)
        embedded_inputs = torch.cat([first_inputs.unsqueeze(1), embedded_targets[:,:-1,:]], dim=1)

        for i in range(embedded_inputs.shape[1]):
        # for input_ in embedded_inputs:
            input_ = embedded_inputs[:, i, :]
            if evaluation_mode or random.random() < tf_thresh:
            # if False:
                input_ = self.embedding(predicted_outputs[-1])
            current_h = self.gru_cell(input_, current_h)
            # weights = current_h @ self.W_a(attention_tensor).squeeze().T
            weights = current_h.unsqueeze(1).bmm(self.W_a(attention_tensor).transpose(1,2)).squeeze(1)
            probs = nn.Softmax(dim=-1)(weights/(self.hidden_size**0.5)) #normalization like in Attention is all you need
            # attentioned = probs @ attention_tensor.squeeze()
            attentioned = probs.unsqueeze(1).bmm(attention_tensor).squeeze(1)
            with_hidden = torch.cat([current_h, attentioned], dim=-1)
            current_output = self.W_s(with_hidden)
            outputs.append(current_output)
            predicted_outputs.append(torch.argmax(current_output, dim=-1))


        outputs = torch.stack(outputs, dim=1)
        return outputs

