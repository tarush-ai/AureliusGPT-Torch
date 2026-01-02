import os, torch, torch.nn as nn
from model.preprocess import Preprocessor
from model.tokenizer import Tokenizer
from model.model import Transformer
from config import PROJECT_ROOT, batch_size, num_epochs, lr

class Train:
    def __init__(self):
        self.pre = Preprocessor()
        self.tokenizer = Tokenizer()
        self.model = Transformer()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.weight_path = os.path.join(PROJECT_ROOT, "data", "weights.pt")


    def process_and_train_tokenizer(self):
        with open(os.path.join(PROJECT_ROOT, "meditations.txt"), "r") as f:
            meditations = f.read()
        
        processed = self.pre.process(meditations)
        self.tokenizer.train(processed[1])
        self.tokenized = self.tokenizer.encode(processed[0])
    
    def seperate_into_batches(self):
        seperated = []
        for i in range(0, len(self.tokenized), batch_size):
            chunked = self.tokenized[i:i+batch_size]
            seperated.append(chunked)
            padding_number = batch_size - len(seperated[-1])
            if padding_number > 0:
                chunked.extend([self.tokenizer.encode("<PAD>")[0]] * padding_number)
        return seperated

    def train_batch(self, batch, print_loss=False):
        batch = torch.LongTensor(batch)
        X = batch[0:-1]
        Y = batch[1:]
        
        logits = self.model.forward(X)
        ce_loss = self.loss(logits, Y)
        self.optimizer.zero_grad()
        ce_loss.backward()
        self.optimizer.step()

        if print_loss:
            print("CE loss is: " + str(ce_loss))

    def train(self):
        self.process_and_train_tokenizer()
        seperated = self.seperate_into_batches()
        for i in range(num_epochs):
            for num, j in enumerate(seperated):
                if num%500 == 0:
                    self.train_batch(j, True)
                else:
                    self.train_batch(j)
            if (i+1)%10 == 0:
                torch.save(self.model.state_dict(), os.path.join(PROJECT_ROOT, "data", f"epoch_{i+1}.pt"))

train = Train()
train.train()