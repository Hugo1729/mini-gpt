import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.module):
    def __init__(self):
        super.__init__()

        #TODO


#masked multi-head attention
#attention = scaled dot-product attention
class CausalSelfAttention(nn.module):
    def __init__(self, n_embd, n_heads, block_sz):
        super.__init__()

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

        attn = (q @ k.transpose(2,3)) / np.sqrt(head_sz)
        attn = attn.masked_fill(self.mask[:seq_len,:seq_len], float('-inf'))
        attn = F.softmax(attn)    # (batch_sz, n_heads, seq_len, seq_len)

        attn = attn @ v  # (batch_sz, n_heads, seq_len, head_sz)

        attn = attn.transpose(1,2)
        attn = attn.view(batch_sz, seq_len, n_embd)

        return self.o_linear(attn)







