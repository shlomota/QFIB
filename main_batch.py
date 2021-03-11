from gru_decoder_batch import DecoderSimple
from gru_attn_decoder import DecoderAttention
from gru_encoder import EncoderRNN
import random
import torch
from torch import nn
from torch import optim
import pandas as pd
import joblib
from preprocess import DATASET_PATH, VOCAB_PATH, PAD_CHAR
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from vocabulary import Vocabulary
import os
import numpy as np

BATCH_SIZE = 32

device = "cuda:0"

def save_model(enc, dec, epoch, postfix=''):
    torch.save(enc, os.path.join("parameters", f'enc_{epoch}{postfix}.pt'))
    torch.save(dec, os.path.join("parameters", f'dec_{epoch}{postfix}.pt'))


def plot_accuracies(train_accs, dev_accs, model_name):

    plt.clf()
    plt.title(model_name)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.plot(train_accs, label='train')
    plt.plot(dev_accs, label='dev')

    plt.xticks(range(len(train_accs)), range(1, len(train_accs) + 1))
    plt.yticks(np.around(np.linspace(0.0, 1.0, num=11), decimals=1))

    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("figures", f'{model_name}.png'))

    plt.show()


def plot_loss(loss, model_name):
    plt.clf()
    plt.title(model_name)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(loss)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("figures", f'{model_name}_loss.png'))
    plt.show()



def evaluate(test_set, enc, dec, print_sentences=True):
    total = 0
    correct = 0

    with torch.no_grad():
        num_batches = int(np.ceil(len(test_set) / BATCH_SIZE))
        # random.shuffle(train_set)

        for i in tqdm(range(num_batches)):
            # x, y = train_set[["x", "y"]].values[:,:]
            x_batch = test_set["x"].values[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            y_batch = test_set["y"].values[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            input_tensor = torch.tensor(np.stack(x_batch)).to(device).long()
            target_tensor = torch.tensor(np.stack(y_batch)).to(device).long()

            encoder_outputs, encoder_h_m = enc(input_tensor)
            decoder_hidden = torch.transpose(encoder_h_m, 0, 1)


            decoder_outputs = dec(target_tensor, decoder_hidden,
                                  enc_outputs=encoder_outputs, enc_input=input_tensor,
                                  evaluation_mode=True)
            decoded_indices = torch.argmax(decoder_outputs, dim=-1)

            # decoded_indices = torch.tensor([dec.vocab.w2i[" "]] * len(decoded_indices.flatten())).to(device)
            # correct += sum(decoded_indices.flatten() == target_tensor.flatten()).item()
            correct += sum(torch.logical_and(decoded_indices.flatten() == target_tensor.flatten(), target_tensor.flatten() != dec.vocab.w2i[PAD_CHAR])).item()
            total += sum(target_tensor.flatten() != dec.vocab.w2i[PAD_CHAR]).item()
            # total += len(decoded_indices.flatten())


            if print_sentences and i % 10 == 0:
                input_sentence_text = "".join([dec.vocab.i2w[i] for i in x_batch[0]])
                target_sentence_text = "".join([dec.vocab.i2w[i] for i in y_batch[0]])
                decoded_sentence_text = "".join([dec.vocab.i2w[i] for i in decoded_indices[0].tolist()])
                input_sentence_text = input_sentence_text[:input_sentence_text.find(PAD_CHAR)]
                target_sentence_text = target_sentence_text[:target_sentence_text.find(PAD_CHAR)]
                decoded_sentence_text = decoded_sentence_text[:decoded_sentence_text.find(PAD_CHAR)]
                print(f'input:    {input_sentence_text}')
                print(f'expected: {target_sentence_text}')
                print(f'result:   {decoded_sentence_text}')

    accuracy = correct / total
    return accuracy

def train_batch(input_tensor, target_tensor, enc, dec,
                enc_optimizer, dec_optimizer, criterion):

    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

    # input_length = input_tensor.size(0)
    # target_length = target_tensor.size(0)

    encoder_outputs, encoder_h_m = enc(input_tensor)
    decoder_hidden = torch.transpose(encoder_h_m, 0, 1)
    decoder_outputs = dec(target_tensor, decoder_hidden, enc_input=input_tensor, enc_outputs=encoder_outputs)

    criterion_target = target_tensor.view(-1)

    decoder_outputs = decoder_outputs.view(-1, dec.output_size)

    indices = (criterion_target != dec.vocab.w2i[PAD_CHAR]).nonzero(as_tuple=True)
    loss = criterion(decoder_outputs[indices], criterion_target[indices])
    loss.backward()

    enc_optimizer.step()
    dec_optimizer.step()

    return loss.item() / (decoder_outputs.shape[0])


def train(n_epochs, train_set, dev_set, enc, dec, criterion,
          patience=-1, save=True, print_sentences=False):

    train_accuracies = []
    losses = []
    epoch_loss = 0  # Reset every epoch

    dev_accuracies = []
    best_dev_accuracy = 0
    epochs_without_improvement = 0  # for early stopping (patience)

    encoder_optimizer = optim.Adam(enc.parameters())
    decoder_optimizer = optim.Adam(dec.parameters())

    for epoch in range(1, n_epochs + 1):
        train_set = train_set.sample(frac=1)
        num_batches = int(np.ceil(len(train_set) / BATCH_SIZE))
        # random.shuffle(train_set)

        for i in tqdm(range(num_batches)):
            # x, y = train_set[["x", "y"]].values[:,:]
            x_batch = train_set["x"].values[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            y_batch = train_set["y"].values[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            x_batch = torch.tensor(np.stack(x_batch)).to(device).long()
            y_batch = torch.tensor(np.stack(y_batch)).to(device).long()

            loss = train_batch(x_batch, y_batch, enc, dec,
                               encoder_optimizer, decoder_optimizer, criterion)
            epoch_loss += loss

        # for x, y in tqdm(train_set[["x", "y"]].values):
        #     # input_tensor = enc.vocab.sentence_to_tensor(x)
        #     # target_tensor = dec.vocab.sentence_to_tensor(y)
        #     input_tensor = torch.tensor(x).to(device).unsqueeze(1)
        #     target_tensor = torch.tensor(y).to(device).unsqueeze(1)
        #     loss = train_single_example(input_tensor, target_tensor, enc, dec,
        #                                 encoder_optimizer, decoder_optimizer, criterion)
        #     epoch_loss += loss

        average_loss = epoch_loss / len(train_set)
        losses.append(average_loss)
        epoch_loss = 0

        train_accuracy = evaluate(train_set, enc, dec, print_sentences)
        train_accuracies.append(train_accuracy)

        dev_accuracy = evaluate(dev_set, enc, dec, print_sentences)
        dev_accuracies.append(dev_accuracy)

        print(f'Epoch #{epoch}:\n'
              f'Loss: {average_loss:.7f}\n'
              f'Train accuracy: {train_accuracy:.6f}\n'
              f'Dev accuracy: {dev_accuracy:.6f}')
              # f'Time elapsed (remaining): {timeSince(start, epoch / n_epochs)}')


        if save:
            save_model(enc, dec, epoch)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == patience:
                print(f'Training didn\'t improve for {patience} epochs\n'
                      f'Stopped Training at epoch {epoch}\n'
                      f'Best epoch: {epoch - patience}')
                break

    return losses, train_accuracies, dev_accuracies




dataset = pd.read_json(DATASET_PATH)
dataset = dataset.sample(frac=0.1)
train_sentences, dev_sentences = train_test_split(dataset)
vocab = joblib.load(VOCAB_PATH)
# general settings, to be used with all the models
enc_input_size = 256
enc_hidden_size = 256

dec_input_size = 256
dec_hidden_size = 256

n_epochs = 40
do_print = False


enc1 = EncoderRNN(enc_input_size, enc_hidden_size, vocab, device=device).to(device)
# dec1 = DecoderSimple(dec_input_size, dec_hidden_size, vocab, device=device).to(device)
dec1 = DecoderAttention(dec_input_size, dec_hidden_size, vocab, device=device).to(device)
criterion1 = nn.CrossEntropyLoss()

losses1, train_accs1, dev_accs1 = train(n_epochs, train_sentences, dev_sentences,
                                        enc1, dec1, criterion1, print_sentences=do_print)
plot_accuracies(train_accs1, dev_accs1, 'attention_decoder')
plot_loss(losses1, 'attention_decoder')
