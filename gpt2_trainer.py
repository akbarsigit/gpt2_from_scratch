from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


@dataclass
class GPT2Config:
    block_size: int = 1024 # max length of the sequence
    vocab_size: int = 50257 # number of tokens in the vocabulary. 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers in the transformer (hidden layers)
    n_head: int = 12 # number of heads in the multiheadattention models
    n_embd: int = 768 # embedding dimension of the model


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__() 
        assert config.n_embd % config.n_head == 0 # make sure that the n_embd is divisible by n_head
        # key, query, value projections for all heads, but in batch
        # the shape of the weight is (n_embd, n_embd * 3) because we need to project the key, query, and value
        self.c_atten = nn.Linear(config.n_embd, config.n_embd * 3) # (C, 3C)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # (C, C)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # doing tril as an 'bias'. This not really 'bias', but more of a mask. But following the naming 
        # of OpenAI/HF.
        # here, we create masking and rearange it as an 4 dim, with (1, 1, T, T) shape
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # get the batch size, sequence length, and the embedding dimension (n_embd (C))

        qkv = self.c_atten(x) # project the input to key, query, and value => (B, T, 3C)
        q, k, v = qkv.split(C, dim=2) # split the key, query, and value => (B, T, C) each
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) => split the key to n_head => (B, n_head, T, C // n_head) => (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) => split the query to n_head => (B, n_head, T, C // n_head) => (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) => split the value to n_head => (B, n_head, T, C // n_head) => (B, nh, T, hs)

        # here, we calculate the attention score
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, hs) @ (B, nh, hs, T) => (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # mask the attention score => (B, nh, T, T)
        att = F.softmax(att, dim=-1) # softmax the attention score => (B, nh, T, T)

        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) => (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, nh, T, hs) => (B, T, nh, hs) => (B, T, C)
        # output projection
        y = self.c_proj(y) # project the output to the original embedding dimension => (B, T, C)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        # this approximate version is the one used in the original GPT-2 because
        # when implemented using tensorflow there is error function that is slow in tensorflow
        # but right now, the version of gelu in pytorch is fast enough and does not have numerical issues.

        # reason using gelu is because using relu will cause the output to be zero if the input is negative
        # and using leaky relu will cause the output to be negative if the input is negative
        # so, using gelu is the best option because it will not cause the output to be zero if the input is negative
        # and it allow the gradient to flow
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

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
        return x

# =========================GPT2=========================

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

    # def forward(self, x):
    #     # note: x here was tokenized, and the shape is (B, T). All possible tokens are in the range of vocab_size
    #     # that we have the representation of it in the wte embedding layer
    #     B, T = x.shape
    #     wte = self.transformer['wte'](x) # lookup the token in the embedding layer => (B, T, n_embd/C)
    #     wpe = self.transformer['wpe'](torch.arange(T, device=x.device)) # lookup the position in the embedding layer => (T, n_embd/C)
    #     h = wte + wpe # sum the token and position embedding => (B, T, C) + (T, C) => (B, T, C) + (B, T, C) => (B, T, C) // kindda like summation of the C dimension
    #     for block in self.transformer['h']:
    #         h = block(h)
    #     h = self.transformer['ln_f'](h) # normalize the output of the last block over the C dim => (B, T, C)
    #     lm_logits = self.lm_head(h) # linear layer to project the output of the last block to the vocab_size => (B, T, vocab_size)
    #     return lm_logits.view(*input_shape, -1)