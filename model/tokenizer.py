import os
import sentencepiece as spm
from config import PROJECT_ROOT, vocab_size

class Tokenizer:
    def __init__(self):
        self.path = os.path.join(PROJECT_ROOT, "data", "tokenizer.model")
    
    def train(self, text):
        spm.sentencePieceTrainer.train(input=text, model_prefix="tokenizer", vocab_size=vocab_size, user_defined_symbols=["<BEGIN>", "<END>", "<PAD>"])
        self.sp = spm.SentencePieceProcessor(model_file=self.path)
    
    def encode(self, text):
        return self.sp.encode(text)