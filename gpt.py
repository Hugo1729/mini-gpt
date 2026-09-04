import torch
from torch import nn

class MLP(nn.module):
    def __init__(self):
        super.__init__()

        #TODO


#scaled dot product masked (causal) self attention
class CausalSelfAttention(nn.module):
    def __init__(self, n_embd, n_heads):
        super.__init__()

        assert n_embd % n_heads == 0
        self.n_embd = n_embd
        self.n_heads = n_heads

        self.q_linear = nn.Linear(n_embd, n_embd, bias=False)
        self.k_linear = nn.Linear(n_embd, n_embd, bias=False)
        self.v_linear = nn.Linear(n_embd, n_embd, bias=False)


    def forward(self, X):
        batch_sz, seq_len, n_embd = X.size()

        assert n_embd == self.n_embd

        q = self.q_linear(X)
        k = self.k_linear(X)
        v = self.v_linear(X)

        head_sz = self.n_embd // self.n_heads

        




