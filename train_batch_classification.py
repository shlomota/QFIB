from gru_decoder_batch import DecoderSimple
from gru_attn_decoder import DecoderAttention
from gru_encoder import EncoderRNN
from gru_model import GRUModel
import random
import torch
from torch import nn
from torch import optim
import pandas as pd
import joblib
from preprocess import DATASET_PATH, VOCAB_PATH, PAD_CHAR, MASK_CHAR
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from vocabulary import Vocabulary
import os
import numpy as np
import sys

BATCH_SIZE = 32

device = "cuda:0"
def my_print(str):
    sys.stdout.write(str + "\n")

def save_model(enc, epoch, postfix=''):
    torch.save(enc, os.path.join("parameters", f'enc_{epoch}{postfix}.pt'))


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

    # plt.show()


def plot_loss(loss, model_name):
    plt.clf()
    plt.title(model_name)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(loss)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("figures", f'{model_name}_loss.png'))
    # plt.show()



def evaluate(test_set, enc, print_sentences=True):
    total = 0
    correct = 0

    with torch.no_grad():
        num_batches = int(np.ceil(len(test_set) / BATCH_SIZE))
        # random.shuffle(train_set)

        for i in range(num_batches):
            # x, y = train_set[["x", "y"]].values[:,:]
            x_batch = test_set["x"].values[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            y_batch = test_set["y"].values[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            input_tensor = torch.tensor(np.stack(x_batch)).to(device).long()
            target_tensor = torch.tensor([element for sublist in y_batch for element in sublist]).to(device).long()

            encoder_outputs, encoder_h_m = enc(input_tensor)
            logits = encoder_outputs[torch.where(input_tensor==enc.vocab.w2i[MASK_CHAR])]
            preds = torch.argmax(logits, dim=-1)
            a=5

            correct += sum(preds == target_tensor).item()
            total += len(target_tensor)


            if print_sentences and i % 10 == 0:
                input_sentence_text = "".join([enc.vocab.i2w[i] for i in x_batch[0]])
                target_sentence_text = "".join([enc.vocab.i2w[i] for i in y_batch[0]])
                decoded_sentence_text = "".join([enc.vocab.i2w[i] for i in encoder_outputs[0][torch.where(input_tensor[0]==enc.vocab.w2i[MASK_CHAR])].tolist()])
                input_sentence_text = input_sentence_text[:input_sentence_text.find(PAD_CHAR)]
                target_sentence_text = target_sentence_text[:target_sentence_text.find(PAD_CHAR)]
                decoded_sentence_text = decoded_sentence_text[:decoded_sentence_text.find(PAD_CHAR)]
                my_print(f'input:    {input_sentence_text}')
                my_print(f'expected: {target_sentence_text}')
                my_print(f'result:   {decoded_sentence_text}')

    accuracy = correct / total
    return accuracy

def train_batch(input_tensor, targets, enc, enc_optimizer, criterion):

    enc_optimizer.zero_grad()
    # input_length = input_tensor.size(0)
    # target_length = target_tensor.size(0)

    encoder_outputs, encoder_h_m = enc(input_tensor)
    preds = encoder_outputs[torch.where(input_tensor==enc.vocab.w2i[MASK_CHAR])]

    loss = criterion(preds, targets)
    loss.backward()
    enc_optimizer.step()
    return loss.item() / (preds.shape[0])


def train(n_epochs, train_set, dev_set, enc, criterion,
          patience=-1, save=True, print_sentences=False):

    train_accuracies = []
    losses = []
    epoch_loss = 0  # Reset every epoch

    dev_accuracies = []
    best_dev_accuracy = 0
    epochs_without_improvement = 0  # for early stopping (patience)

    encoder_optimizer = optim.Adam(enc.parameters())

    for epoch in tqdm(range(1, n_epochs + 1)):
        train_set = train_set.sample(frac=1)
        num_batches = int(np.ceil(len(train_set) / BATCH_SIZE))

        for i in range(num_batches):
            # x, y = train_set[["x", "y"]].values[:,:]
            x_batch = train_set["x"].values[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            y_batch = train_set["y"].values[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            x_batch = torch.tensor(np.stack(x_batch)).to(device).long()
            y_batch = torch.tensor([element for sublist in y_batch for element in sublist]).to(device).long()
            loss = train_batch(x_batch, y_batch, enc,
                               encoder_optimizer, criterion)
            epoch_loss += loss

        # for x, y in tqdm(train_set[["x", "y"]].values):
        #     # input_tensor = enc.vocab.sentence_to_tensor(x)
        #     # target_tensor = enc.vocab.sentence_to_tensor(y)
        #     input_tensor = torch.tensor(x).to(device).unsqueeze(1)
        #     target_tensor = torch.tensor(y).to(device).unsqueeze(1)
        #     loss = train_single_example(input_tensor, target_tensor, enc, dec,
        #                                 encoder_optimizer, decoder_optimizer, criterion)
        #     epoch_loss += loss

        average_loss = epoch_loss / len(train_set)
        losses.append(average_loss)
        epoch_loss = 0

        train_accuracy = evaluate(train_set, enc, print_sentences)
        train_accuracies.append(train_accuracy)

        dev_accuracy = evaluate(dev_set, enc, print_sentences)
        dev_accuracies.append(dev_accuracy)

        my_print(f'Epoch #{epoch}:\n'
              f'Loss: {average_loss:.7f}\n'
              f'Train accuracy: {train_accuracy:.6f}\n'
              f'Dev accuracy: {dev_accuracy:.6f}')
              # f'Time elapsed (remaining): {timeSince(start, epoch / n_epochs)}')


        if save:
            save_model(enc, epoch)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == patience:
                my_print(f'Training didn\'t improve for {patience} epochs\n'
                      f'Stopped Training at epoch {epoch}\n'
                      f'Best epoch: {epoch - patience}')
                break

    return losses, train_accuracies, dev_accuracies


if __name__ == "__main__":
    # dataset = pd.read_json(r"C:\Users\soki\PycharmProjects\QFIB\data\dataset_only_mask_y.json")
    # dataset = pd.read_json(r"data/dataset_only_mask_y.json")
    dataset = pd.read_json(r"data/full_dataset_only_mask_y.json")
    # dataset = dataset.sample(frac=0.1)
    train_sentences, dev_sentences = train_test_split(dataset)
    vocab = joblib.load(VOCAB_PATH)
    # general settings, to be used with all the models
    enc_input_size = 512
    enc_hidden_size = 512

    n_epochs = 40
    # n_epochs = 5
    do_print = False

    # enc1 = EncoderRNN(enc_input_size, enc_hidden_size, vocab, device=device).to(device)
    model = GRUModel(enc_input_size, enc_hidden_size, len(vocab.w2i), vocab, device=device).to(device)
    criterion1 = nn.CrossEntropyLoss()

    losses1, train_accs1, dev_accs1 = train(n_epochs, train_sentences, dev_sentences,
                                            model, criterion1, print_sentences=do_print)
    plot_accuracies(train_accs1, dev_accs1, 'gru_clf')
    plot_loss(losses1, 'gru_clf')

