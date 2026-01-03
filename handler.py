import sys
import os
import torch
import re
from typing import Dict, List, Any

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from model.model import Transformer
from model.vocab.tokenizer import Tokenizer
import config

class EndpointHandler():
    def __init__(self, path=""):
        model_dir = path if path else repo_root
        model_file = os.path.join(model_dir, "data", "epoch_10.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.model = Transformer().to(self.device)
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.eval()
        self.tokenizer = Tokenizer() 

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        inputs = data.pop("inputs", data)
        encoded = torch.LongTensor(self.tokenizer.encode(inputs)).to(self.device)
        
        currtoken = ""
        outputstring = ""
        countcheck = 0
        
        while currtoken != "<END>" and countcheck < config.max_tokens:
            with torch.no_grad(): 
                currtoken_out = self.model(encoded)
                last_token_logits = currtoken_out[-1, :] 

                if config.argmax:
                    predictions = torch.argmax(last_token_logits).item()
                else:
                    logits = last_token_logits / config.temperature
                    probs = torch.softmax(logits, dim=-1)
                    predictions = torch.multinomial(probs, num_samples=1).item()

                currtoken = self.tokenizer.decode([predictions]).strip()
                if re.match(r'^[.,!?;:]', currtoken):
                    if outputstring.endswith(" "):
                        outputstring = outputstring[:-1]
                    outputstring += currtoken + " "
                else:
                    outputstring += currtoken + " "
                
                next_token = torch.tensor([predictions], dtype=torch.long, device=self.device)
                encoded = torch.cat((encoded, next_token), dim=0)
                if encoded.shape[0] > config.max_seq_length:
                    encoded = encoded[-config.max_seq_length:]
            
            countcheck += 1
            
        #postprocesses
        text = re.sub("<BEGIN>", "\n\n", outputstring)
        text = re.sub("<END>", "\n\n", text)
        text = "AURELIUS: " + text
        
        return [{"generated_text": text}]

