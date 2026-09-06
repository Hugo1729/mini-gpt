import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, n_embd, n_hidden):
        super().__init__()

        self.linear1 = nn.Linear(n_embd, n_hidden)
        self.activation = nn.ELU()
        self.linear2 = nn.Linear(n_hidden, n_embd)

    def forward(self, X):
        X = self.linear1(X)
        X = self.activation(X)
        X = self.linear2(X)

        return X


#masked multi-head attention
#attention = scaled dot-product attention
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads, block_sz):
        super().__init__()

        assert n_embd % n_heads == 0
        self.n_embd = n_embd
        self.n_heads = n_heads

        self.q_linear = nn.Linear(n_embd, n_embd, bias=False)
        self.k_linear = nn.Linear(n_embd, n_embd, bias=False)
        self.v_linear = nn.Linear(n_embd, n_embd, bias=False)

        self.o_linear = nn.Linear(n_embd, n_embd, bias=False)

        self.mask = torch.ones(block_sz, block_sz, dtype=torch.bool).triu(diagonal=1)

    def forward(self, X):
        batch_sz, seq_len, n_embd = X.size()

        assert n_embd == self.n_embd

        head_sz = self.n_embd // self.n_heads

        q = self.q_linear(X)
        k = self.k_linear(X)
        v = self.v_linear(X)

        for z in [q, k, v]:
            z = z.view(batch_sz, seq_len, self.n_heads, head_sz)
            z = z.transpose(1,2)     # (batch_sz, n_heads, seq_len, head_sz)

        attn = (q @ k.transpose(2,3)) / torch.sqrt(head_sz)
        attn = attn.masked_fill(self.mask[:seq_len,:seq_len], float('-inf'))
        attn = F.softmax(attn)    # (batch_sz, n_heads, seq_len, seq_len)

        attn = attn @ v  # (batch_sz, n_heads, seq_len, head_sz)

        attn = attn.transpose(1,2)
        attn = attn.view(batch_sz, seq_len, n_embd)

        return self.o_linear(attn)


class LayerNorm(nn.Module):
    def __init__(self, epsilon, n_embd):
        super().__init__()

        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))

    def forward(self, X):
        batch_sz, seq_len, n_embd = X.size()

        mu = torch.mean(X, -1, keepdim=True)
        var = torch.var(X, -1, keepdim=True, unbiased=False)

        Z =  (X - mu) / torch.sqrt(var + self.epsilon)

        return self.beta * Z + self.gamma


class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_sz, n_hidden, epsilon):
        super().__init__()

        self.norm1 = LayerNorm(epsilon, n_embd)
        self.attention = CausalSelfAttention(n_embd, n_heads, block_sz)

        self.norm2 = LayerNorm(epsilon, n_embd)
        self.mlp = MLP(n_embd, n_hidden)

    def forward(self, X):
        Z = self.norm1(X)
        X = X + self.attention(Z)

        Z = self.norm2(X)
        X = X + self.mlp(Z)

        return X

class GPT(nn.Module):
    def __init__(self, n_embd, n_heads, block_sz, n_hidden, epsilon, n_tokens, n_blocks):
        super().__init__()

        self.embed = nn.Linear(n_tokens, n_embd)

        self.blocks = [Block(n_embd, n_heads, block_sz, n_hidden, epsilon) for _ in range(n_blocks)]

        self.unembed = nn.Linear(n_embd, n_tokens)

    def forward(self, X):
        X = self.embed(X)

        #TODO positional encoding

        for block in self.blocks:
            X = block(X)

        X = self.unembed(X)

        return X
