import os, torch, torch.nn as nn
import random
from model.datageneration.syntheticformatting import SyntheticFormatter
from model.vocab.preprocess import Preprocessor
from model.vocab.tokenizer import Tokenizer
from model.model import Transformer
from config import PROJECT_ROOT, batch_size, num_epochs, lr

class Train:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")
        
        self.formatter = SyntheticFormatter()
        self.pre = Preprocessor()
        self.tokenizer = Tokenizer()
        self.model = Transformer().to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        
        if os.path.exists(os.path.join(PROJECT_ROOT, "epoch_10.pt")):
             self.save_dir = PROJECT_ROOT
        else:
             self.save_dir = os.path.join(PROJECT_ROOT, "data")
             if not os.path.exists(self.save_dir):
                 os.makedirs(self.save_dir)

        self.weight_path = os.path.join(self.save_dir, "weights.pt")
        
        # Placeholders for data
        self.train_tokenized = []
        self.val_tokenized = []

    def generate_synthetic_meditations(self):
        with open(os.path.join(PROJECT_ROOT, "meditations.txt"), "r") as f:
            meditations = f.read()
        processed = self.formatter.process(meditations)

        with open(os.path.join(PROJECT_ROOT, "meditationssynthetic.txt"), "w") as f:
            f.write(processed[0])

        #Here is where we would use processed meditations (no <BEGIN> or <END> tags)
        #to run Unsloth LoRA on LLAMA 3.2 1B to generate 100k parameters of synthetic Meditations
        #adjacent stoic text. The adapted Colab Notebook is at model/llama321b_meditations_lora_unsloth.py

    def process_and_train_tokenizer(self):
        with open(os.path.join(PROJECT_ROOT, "meditations.txt"), "r") as f:
            meditations = f.read()
        
        meditations_synthetic = ""
        if os.path.exists(os.path.join(PROJECT_ROOT, "meditationssynthetic.txt")):
            with open(os.path.join(PROJECT_ROOT, "meditationssynthetic.txt"), "r") as f:
                meditations_synthetic = f.read()

        processed_pure = self.pre.process(meditations)
        processed_synthetic = self.pre.process(meditations_synthetic)

        processed_overall_sentences = processed_pure[1] + processed_synthetic[1]
        
        tokenizer_path = os.path.join(self.save_dir, "tokenizer")
        self.tokenizer.train(processed_overall_sentences, model_prefix=tokenizer_path)
        
        self.trained, self.val = self.train_val(processed_pure[0], processed_synthetic[0])
        self.train_tokenized = self.tokenizer.encode(self.trained)
        self.val_tokenized = self.tokenizer.encode(self.val)

    def train_val(self, pure, synthetic):
        eighty = 4 * len(pure) // 5
        val = pure[eighty:]
        train_pure = pure[:eighty]
        trained = train_pure + train_pure + synthetic
        return trained, val

    def seperate_into_batches(self, token_list):
        seperated = []
        for i in range(0, len(token_list), batch_size):
            chunked = token_list[i:i+batch_size]
            padding_number = batch_size - len(chunked)
            if padding_number > 0:
                chunked.extend([self.tokenizer.encode("<PAD>")[0]] * padding_number)
            seperated.append(chunked)
        return seperated

    def train_batch(self, batch, print_loss=False):
        batch = torch.LongTensor(batch).to(self.device)
        X = batch[0:-1]
        Y = batch[1:]
        
        logits = self.model.forward(X)
        ce_loss = self.loss(logits.view(-1, logits.size(-1)), Y.view(-1))
        self.optimizer.zero_grad()
        ce_loss.backward()

        if print_loss:
            print("CE loss is: " + str(ce_loss))

        #calculating gradient norm for train/val overfitting check
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.optimizer.step()
        return ce_loss.item(), total_norm

    def evaluate(self, val_batches):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_batches:
                batch_tensor = torch.LongTensor(batch).to(self.device)
                X = batch_tensor[0:-1]
                Y = batch_tensor[1:]
                logits = self.model.forward(X)
                loss = self.loss(logits, Y)
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(val_batches) if val_batches else 0

    def train(self):
        self.process_and_train_tokenizer()

        train_batches = self.seperate_into_batches(self.train_tokenized)
        val_batches = self.seperate_into_batches(self.val_tokenized)    
        
        for i in range(num_epochs):
            random.shuffle(train_batches)

            epoch_loss = 0
            epoch_grad_norm = 0

            for j, batch in enumerate(train_batches):
                loss, grad_norm = self.train_batch(batch)
                epoch_loss += loss
                epoch_grad_norm += grad_norm
                
                if j % 500 == 0 and j > 0:
                    print(f"  Batch {j}; Loss: {loss:.4f}; Grad Norm: {grad_norm:.4f}")


            avg_train_loss = epoch_loss / len(train_batches)
            avg_grad_norm = epoch_grad_norm / len(train_batches)
            val_loss = self.evaluate(val_batches)

            print(f"Epoch {i+1}; Train Loss: {avg_train_loss:.4f}; Val Loss: {val_loss:.4f}; Train Grad Norm: {avg_grad_norm:.4f}")
        
            if (i+1) % 10 == 0:
                save_path = os.path.join(self.save_dir, f"epoch_{i+1}.pt")
                torch.save(self.model.state_dict(), save_path)


if __name__ == "__main__":
    train = Train()
    #train.generate_synthetic_meditations() #uncomment to generate synthetic meditations, then comment out the next line
    train.train()
