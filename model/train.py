import os
from preprocess import Preprocessor
from tokenizer import Tokenizer
from model import Transformer
from config import PROJECT_ROOT, batch_size, num_epochs

class Train:
    def __init__(self):
        self.pre = Preprocessor()
        self.tokenizer = Tokenizer()
        self.model = Transformer()

    def process_and_train_tokenizer(self):
        with open(os.path.join(PROJECT_ROOT, "model", "meditations.txt"), "r") as f:
            meditations = f.read()
        
        processed = self.pre.process(meditations)
        self.tokenizer.train(processed[1])
        self.tokenized = self.tokenizer.encode(processed[0])
    
    def seperate_into_batches(self):
        seperated = []
        for i in range(0, len(self.tokenized), batch_size):
            chunked = self.tokenized[i:i+batch_size]
            seperated.append(chunked)
        while len(seperated[-1]) < batch_size:
            seperated[-1].append(self.tokenizer.encode("<PAD>")[0])
        return seperated

    def train_batch(self, batch):
        #this will require torch
        pass

    def train(self):
        self.process_and_train_tokenizer()
        seperated = self.seperate_into_batches()
        for i in range(num_epochs):
            for j in seperated:
                self.train_batch(j)


