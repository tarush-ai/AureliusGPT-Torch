import os, torch
import torch.nn as nn
from config import PROJECT_ROOT, num_epochs, max_tokens
from model.model import Transformer
from model.tokenizer import Tokenizer

class Run:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model = Transformer()
        self.path = os.path.join(PROJECT_ROOT, "data", f"epochs_{num_epochs}.pt")
        self.model.load_state_dict(torch.load(self.path)) 
        self.model.eval()

    def run(self):
        input_query = input("User: ")
        print("\n")
        encoded = self.tokenizer.encode(input_query)
        encoded = torch.Tensor(encoded)
        output = ""
        countcheck = 0
        while output != "<END>" and countcheck < max_tokens:
            with torch.no_grad(): 
                output = self.model(encoded)
                predictions = torch.argmax(output, dim=1)
                print(f"AureliusGPT: {predictions.item()}")
                print("\n")
        output = predictions.item()
            


