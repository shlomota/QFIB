class Vocabulary(object):
    def __init__(self, vocab):
        self.w2i = vocab
        self.i2w = {v:k for k, v in vocab.items()}