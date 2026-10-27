import torch
from torch import nn
import torch.nn.functional as F
from gpt import PositionalEncoding

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         bias=False,
                                         batch_first=True
        )

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4*embed_dim)
        self.activation = nn.ELU()
        self.linear2 = nn.Linear(4*embed_dim, embed_dim)

    def forward(self, x):
        y = self.layer_norm1(x)

        seq_len = x.shape[1]

        attn_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=seq_len, 
            device=y.device, 
            dtype=y.dtype
        )
        attn_out, _ = self.mha(key=y,
                     query=y,
                     value=y,
                     is_causal=True,
                     attn_mask=attn_mask
        )

        x = x + attn_out

        y = self.layer_norm2(x)
        y = self.linear1(y)
        y = self.activation(y)
        x = x + self.linear2(y)

        return x

class GPT(nn.Module):
    def __init__(self, embed_dim, num_embeddings, num_heads, block_sz, num_blocks):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=num_embeddings,
                                  embedding_dim=embed_dim
        )
        self.pe = PositionalEncoding(n_embd=embed_dim, block_sz=block_sz)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_embeddings)

    def forward(self, x):
        x = self.embed(x)
        x = x + self.pe(x)

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x)
        x = self.linear(x)

        return x