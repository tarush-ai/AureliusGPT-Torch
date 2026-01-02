import os, torch
import torch.nn as nn
from config import PROJECT_ROOT, num_epochs, max_tokens, temperature
from model.model import Transformer
from model.tokenizer import Tokenizer
from model.preprocess import Preprocessor

class Run:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model = Transformer()
        self.path = os.path.join(PROJECT_ROOT, "data", f"epoch_{num_epochs}.pt")
        self.model.load_state_dict(torch.load(self.path)) 
        self.model.eval()

    def run(self):
        with open(os.path.join(PROJECT_ROOT, "meditations.txt"), "r") as f:
            meditations = f.read()
        
        pre = Preprocessor()
        processed = pre.process(meditations)
        self.tokenizer.train(processed[1])

        print("AureliusGPT\n\n\n\n")

        input_query = input("User: ")
        print("\n")
        encoded = self.tokenizer.encode(input_query)
        encoded = torch.LongTensor(encoded)
        output = ""
        countcheck = 0
        while output != "<END>" and countcheck < max_tokens:
            with torch.no_grad(): 
                output = self.model(encoded)
                last_token_logits = output[-1, :] 

                logits = last_token_logits / temperature
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.multinomial(probs, num_samples=1).item()



                output = self.tokenizer.decode([predictions])
                print(output + " ", flush=True)
                next_token = torch.tensor([predictions], dtype=torch.long)
                encoded = torch.cat((encoded, next_token), dim=0)
        output = predictions.item()
            

if __name__ == "__main__":
    run = Run()
    run.run()
