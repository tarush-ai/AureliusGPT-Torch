import os, torch, re
import torch.nn as nn
from config import PROJECT_ROOT, num_epochs, max_tokens, temperature, max_seq_length, justification_model
from model.model import Transformer
from model.tokenizer import Tokenizer
from model.preprocess import Preprocessor
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Run:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model = Transformer()
        self.path = os.path.join(PROJECT_ROOT, "data", f"epoch_50.pt")
        self.model.load_state_dict(torch.load(self.path)) 
        self.model.eval()

    def run(self):
        with open(os.path.join(PROJECT_ROOT, "meditations.txt"), "r") as f:
            meditations = f.read()
        
        print("AureliusGPT\n\n")

        print("A model trained on Meditations by Marcus Aurelius.\n\n")

        print("Press 'Enter' to stop the conversation.")


        input_query = input("User: ")

        if input_query == "":
            return "INTERACTION_COMPLETE"

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

        return input_query, outputstring
    
    def postprocess(self, text):
        text = re.sub("<BEGIN>", "\n\n", text)
        text = re.sub("<END>", "\n\n", text)
        return "AURELIUS: " + text

    def justify(self, text):
        client = OpenAI()

        prompt = "Your responsibility is to justify the output of a toy model fitted on Meditations by Marcus Aurelius. Please read the user's prompt, the model's response, and give the model a score out of 100 of prediction given it's status as a toy model. Generate a short report of its accuracy, identifying potential semantic, linguistic, and Stoic-meaning based connections in the generation."

        response = client.chat.completions.create(
            model=justification_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )

        print(response.choices[0].message.content)

    def main(self):
        outputstring = ""
        while outputstring != "INTERACTION_COMPLETE":
            ran = self.run()
            outputstring = ran[1]
            postprocessed = self.postprocess(outputstring)
            print(postprocessed)
            #print(self.justify(ran[0] + postprocessed))



if __name__ == "__main__":  
    run = Run()
    run.main()
    
