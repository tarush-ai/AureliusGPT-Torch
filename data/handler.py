import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any
import torch, re
from model.model import Transformer
from model.vocab.tokenizer import Tokenizer
from config import argmax, temperature, max_seq_length, max_tokens

class EndpointHandler():
    def __init__(self, path=""):
        model_dir = path if path else os.path.dirname(os.path.abspath(__file__))
        model_file = os.path.join(model_dir, "epoch_10.pt")
        if not os.path.exists(model_file):
            model_file = "epoch_10.pt"
            
        self.model = Transformer()
        self.model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        self.model.eval()
        self.tokenizer = Tokenizer() #already loads from weights within logic

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        inputs = data.pop("inputs", data)
        
        #encodes
        encoded = torch.LongTensor(self.tokenizer.encode(inputs)) 
        
        #generates
        currtoken = ""
        outputstring = ""
        countcheck = 0
        while currtoken != "<END>" and countcheck < max_tokens:
            with torch.no_grad(): 
                currtoken = self.model(encoded)
                last_token_logits = currtoken[-1, :] 

                if argmax:
                    predictions = torch.argmax(last_token_logits).item()
                else:
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
        
        #postprocesses
        text = re.sub("<BEGIN>", "\n\n", outputstring)
        text = re.sub("<END>", "\n\n", text)
        text = "AURELIUS: " + text

        
        return [{"generated_text": text}]

