from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


@dataclass
class GPTConfig:
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
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # (C, 3C)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # (C, C)

        # flagging for linear layer scaler
        self.c_proj.MYGPT_SCALE_INIT = 1

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

        qkv = self.c_attn(x) # project the input to key, query, and value => (B, T, 3C)
        q, k, v = qkv.split(C, dim=2) # split the key, query, and value => (B, T, C) each

        # nh is the number of heads, hs is the size of the head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) => split the key to n_head => (B, n_head, T, C // n_head) => (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) => split the query to n_head => (B, n_head, T, C // n_head) => (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) => split the value to n_head => (B, n_head, T, C // n_head) => (B, nh, T, hs)

        # here, we calculate the attention score
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, hs) @ (B, nh, hs, T) => (B, nh, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # mask the attention score => (B, nh, T, T)
        # att = F.softmax(att, dim=-1) # softmax the attention score => (B, nh, T, T)
        # y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) => (B, nh, T, hs)

        # we can comment this 4 lines above to call flashattention.
        # basically, the torch.compile cannot optimize the attention mechanism
        # so we use flashattention fused the kernel and do softmax in the streaming manner (online softmax calculation).
        # We also dont need to store the TxT attention matrix in the GPU memory (HBO)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        self.c_proj.MYGPT_SCALE_INIT = 1

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

        # weight sharing (tieing) scheme
        # so the logic behind this is because there will be some semantic meaning between the token embedding
        # and the output layer, so we want to share the weight between the token embedding and the output layer
        # We also saving 768 * 50257 =~ 40M parameters (30% of the model size of 124M)
        self.transformer.wte.weight = self.lm_head.weight

        # initialize the weights
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'MYGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # note: idx here was tokenized, and the shape is (B, T). All possible tokens are in the range of vocab_size
        # that we have the representation of it in the wte embedding layer
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        
        # foward the token and position embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # shape (T, C). note C = n_embd
        tok_emb = self.transformer.wte(idx) # shape (B, T, C)
        x = tok_emb + pos_emb # shape (B, T, C)

        # foward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the last layer norm and the classifier head
        x = self.transformer.ln_f(x) # shape (B, T, C)
        logits = self.lm_head(x) # shape (B, T, vocab_size)

        # note. So the logic here is, if we have input B, T
        # so each B, T, we will calculate every single logits for what token comes next in the sequence.
        # so what is the B, T+1 sequence will be, we will calculate the logits for each token in the vocab_size

        # if we have the target, we will calculate the loss
        loss = None
        if targets is not None:
            # this will be (B*T, vocab_size) F (B*T) => (loss)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from Hugging Face's transformers library"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained GPT: %s" % model_type)

        # n_layer, n_head, and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        # instantiate minGPT model from-scratch
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys() # getting all layers in the model
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the mask / buffer layers as this is just for the attention mask

        # load pretrained model weights from Hugging Face's transformers library
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # transfer pretrained weights from Hugging Face's transformers library to minGPT model
        # copy while ensuring all of the parameters are aliggned and match in names and shapes
        sd_keys_hf = sd_hf.keys()

        # iterate over the layers in the minGPT model and also filter out the mask / buffer layers as its not parameters
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)

        # as some of the weight in the model is transposed in the implementation of the model with tensorflow
        # we need to transpose it back so it can be aligned with the minGPT model with torch
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # print("transposing " + str(k) + " with shape " + str(sd_hf[k].shape) + " to " + str(sd[k].shape))
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    # fill the weight in the minGPT model with the transposed weight from the Hugging Face's transformers library
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # find/start with all of the candidate parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        


# =========================Detecting Device=========================
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Using %s" % device)

# =========================Reproduce Ability=========================
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# # =========================Tokenizing=========================
# enc = tiktoken.get_encoding("gpt2")
# # tokens = enc.encode("Hello, akbar is so cool that")

# # load the datasets to train the model
# with open('input.txt', 'r') as f:
#     text = f.read()

# text = text[:1000]
# tokens = enc.encode(text)

# ===============Data Loader=========================
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
       
        with open('input.txt', 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets => we offset the target by 1

        # move the position
        self.current_position += B * T

        # reset the position if we reach the end of the dataset
        if self.current_position + (B * T + 1) >= len(self.tokens):
            self.current_position = 0
        return x, y


# =================Batching/Creating datasets==================
# create 4 batch of 32 sequence length
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])
# buf = buf.to(device)

# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

# Activating TF32 Precision
torch.set_float32_matmul_precision('high')

# model = GPT.from_pretrained("gpt2") # loading pretrained weights
# model = GPT(GPTConfig()) # random initialization of the model
model = GPT(GPTConfig(vocab_size=50304)) # now we actually increase the vocab with dummy tokens so that it has good numbers. 50304 
                                         # can have many 2^x factors, so it can be breaking down to many factors in cuda computations 
print("Model loaded successfully!")
# model.eval()  #when doing generation, do this
model.to(device)

# what compile is doing is to make the model not run in eager mode, so not layer by layer
# instead, it will look at the whole model and optimize its round trip (read/write) using so called kernel fusion
model = torch.compile(model) # compile for the NN => like GCC for c/c++ code => for speedup



# doing learning rate scheduling
max_lr = 6e-4
min_lr = max_lr * 0.1 # this is 10% of the max lr
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_deacy_iters, return min learning rate
    if it >= max_steps:
        return min_lr
    # 3) in between, use cosine learning rate decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0     
    return min_lr + coeff * (max_lr - min_lr)

# print(loss) # tensor(11.0831, grad_fn=<NllLossBackward0>)
# print(loss.shape) # torch.Size([])

import time

train_loader = DataLoaderLite(B=4, T=32)

# this params is got from GPT3 paper
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    # using bfloat on foward pass => some will be keep on float32 and some, mainly on matmul will be on bfloat16
    # with torch.autocast(device_type=device, dtype=torch.bfloat16): => comment this because dont have supported GPU hardware
    logits, loss = model(x, y)
    loss.backward()

    # global gradient clipping - to prevent gradient explosion
    # basically, we will power of two all the gradient, sum them all, and then sqrt them, or having a norm of 1
    # this is to prevent the gradient to be too big, if we have unlucky iteration that have big loss, so it will not explode
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    # we do this because we want to do learning rate decay scheduling
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

    print(f"step {step:4d} | loss: {loss.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | time: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


import sys; sys.exit(0)



# =========================Running model=========================
num_return_sequences = 5
max_length = 32
# https://tiktokenizer.vercel.app/?model=gpt2
tokens = torch.tensor(tokens, dtype=torch.long) # (9, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# right now x is (B, T) where B = 5, T = 9
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# =========================Generate=========================

while x.size(1) < max_length:
    # foward the model to reproduce the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits of the last token
        logits = logits[:, -1, :]
        # get the probability distribution
        probs = F.softmax(logits, dim=-1)
        # do top k sampling to get the next token of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices becomes (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token randomly from the top k
        ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append the new token to the sequence  
        x = torch.cat((x, xcol), dim=1)

# print the generated sequences
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)


