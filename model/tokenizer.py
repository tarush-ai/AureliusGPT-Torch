import os, io
import sentencepiece as spm
from config import PROJECT_ROOT, vocab_size
from model.preprocess import Preprocessor

class Tokenizer:
    def __init__(self):
        self.path = os.path.join(PROJECT_ROOT, "data", "tokenizer.model")
    
    def train(self, all_sentences):
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(all_sentences), 
            model_prefix="data/tokenizer", 
            vocab_size=vocab_size,
            user_defined_symbols=["<BEGIN>", "<END>", "<PAD>"]
        )
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.path)
    
    def encode(self, text):
        return self.sp.encode(text)

process = Preprocessor()
with open(os.path.join(PROJECT_ROOT, "meditations.txt"), "r") as f:
    meditations = f.read()
p = process.process(meditations)

token = Tokenizer()
token.train(p[1])
print(len(token.encode(p[0])))