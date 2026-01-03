import numpy as np
import torch
import torch.nn as nn
from config import num_blocks, vocab_size, d_model, h, d_head, d_ff, max_seq_length

#main transformer

class Util:
    def sinusoidal(self):
        PE = np.zeros((max_seq_length, d_model))
        
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                div_term = 10000 ** (i / d_model)
                PE[pos, i] = np.sin(pos / div_term)
                if i + 1 < d_model:
                    PE[pos, i + 1] = np.cos(pos / div_term)
                    
        return PE

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock() for i in range(num_blocks)])
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        util = Util()
        self.positionals = util.sinusoidal()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, X):
        embeddings = self.embeddings(X)
        positionals = torch.tensor(self.positionals[:X.shape[0]]).float() 
        embeddings = embeddings + positionals

        for block in self.blocks:
            embeddings = block(embeddings)

        return self.linear(embeddings)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentionblock = AttentionBlock()
        self.layernorm = LayerNorm()
        self.ffn = FFN()
        self.layernorm2 = LayerNorm()
    
    def forward(self, X):
        X = self.layernorm(X + self.attentionblock(X))
        X = self.layernorm2(X + self.ffn(X))
        return X

#attention
class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentionheads = nn.ModuleList([AttentionHead() for i in range(h)])
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, X):
        headoutputs = [head(X) for head in self.attentionheads]
        MHA = torch.cat(headoutputs, dim=-1)
        return self.Wo(MHA)
    

class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.queries = nn.Linear(d_model, d_head, bias=False)
        self.keys = nn.Linear(d_model, d_head, bias=False)
        self.values = nn.Linear(d_model, d_head, bias=False)
        
    def forward(self, X):
        Q = self.queries(X)
        K = self.keys(X)
        V = self.values(X)

        scores = Q @ K.T
        scores /= (d_head ** 0.5)
        mask = torch.tril(torch.ones(X.shape[0], X.shape[0], device=X.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)  
        return attention @ V      

#adding / prenorm
class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
    def forward(self, X):
        return self.norm(X)

#ffn
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(), 
            nn.Linear(d_ff, d_model)
        )

    def forward(self, X):
        return self.net(X)