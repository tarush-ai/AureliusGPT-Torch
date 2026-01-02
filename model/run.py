import os, torch, re
import torch.nn as nn
from config import PROJECT_ROOT, num_epochs, max_tokens, temperature, max_seq_length
from model.model import Transformer
from model.tokenizer import Tokenizer
from model.preprocess import Preprocessor

class Run:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model = Transformer()
        self.path = os.path.join(PROJECT_ROOT, "data", f"epoch_12.pt")
        self.model.load_state_dict(torch.load(self.path)) 
        self.model.eval()

    def run(self):
        with open(os.path.join(PROJECT_ROOT, "meditations.txt"), "r") as f:
            meditations = f.read()
        
        print("AureliusGPT\n\n")

        input_query = input("User: ")
        print("\n")
        encoded = self.tokenizer.encode(input_query)
        encoded = torch.LongTensor(encoded)
        currtoken = ""
        outputstring = ""
        countcheck = 0
        while currtoken != "<END>" and countcheck < max_tokens:
            with torch.no_grad(): 
                currtoken = self.model(encoded)
                last_token_logits = currtoken[-1, :] 

                logits = last_token_logits / temperature
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.multinomial(probs, num_samples=1).item()

                currtoken = self.tokenizer.decode([predictions]).strip()
                if re.match(r'^[.,!?;:]', currtoken):
                    if outputstring.endswith(" "):
                        outputstring = outputstring[:-1]
                    outputstring += currtoken + " "
                else:
                    outputstring += currtoken + " "
                next_token = torch.tensor([predictions], dtype=torch.long)
                encoded = torch.cat((encoded, next_token), dim=0)
                if encoded.shape[0] > max_seq_length:
                    encoded = encoded[-max_seq_length:]
        currtoken = predictions

        return outputstring

    

    def postprocess(self, text):
        text = re.sub("<BEGIN>", "\n\n", text)
        text = re.sub("<END>", "\n\n", text)
        return "AURELIUS: " + text
            

if __name__ == "__main__":
    run = Run()
    outputstring = run.run()
    print(run.postprocess(outputstring))
