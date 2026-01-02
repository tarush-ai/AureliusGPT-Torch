import os, io
import sentencepiece as spm
from config import PROJECT_ROOT, vocab_size
from model.preprocess import Preprocessor

class Tokenizer:
    def __init__(self):
        self.path = os.path.join(PROJECT_ROOT, "data", "tokenizer.model")
        self.sp = spm.SentencePieceProcessor()
        if os.path.exists(self.path):
            self.sp.Load(self.path)
    
    def train(self, all_sentences):
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(all_sentences), 
            model_prefix="data/tokenizer", 
            vocab_size=vocab_size,
            user_defined_symbols=["<BEGIN>", "<END>", "<PAD>"]
        )
        self.sp.load(self.path)
    
    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, ids):
        return self.sp.decode(ids)