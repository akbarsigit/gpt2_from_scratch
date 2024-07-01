from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPT2Config:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class Block(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # here, we deviate from the original implementation of attention is all you need,
        # which we use layer norm before the attention layer and before the mlp.
        # we also not including residual to be normalize either
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # make vocab_size possible lookup table the token
            wpe = nn.Embedding(config.block_size, config.n_embd), # this will create lookup table that the size of block_size (sequence)
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #hidden layers => attention + feedforward
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        # note: x here was tokenized, and the shape is (B, T). All possible tokens are in the range of vocab_size
        # that we have the representation of it in the wte embedding layer
        B, T = x.shape
        wte = self.transformer['wte'](x) # lookup the token in the embedding layer => (B, T, n_embd/C)
        wpe = self.transformer['wpe'](torch.arange(T, device=x.device)) # lookup the position in the embedding layer => (T, n_embd/C)
        h = wte + wpe # sum the token and position embedding => (B, T, C) + (T, C) => (B, T, C) + (B, T, C) => (B, T, C) // kindda like summation of the C dimension
        for block in self.transformer['h']:
            h = block(h)
        h = self.transformer['ln_f'](h) # normalize the output of the last block over the C dim => (B, T, C)
        lm_logits = self.lm_head(h) # linear layer to project the output of the last block to the vocab_size => (B, T, vocab_size)
        return lm_logits.view(*input_shape, -1)