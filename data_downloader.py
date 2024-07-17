"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard => total of 100 shards for 10B tokens dataset

# create the cache directory if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# donwlod the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc['text'])) # get the text column from the dataset
    tokens_np = np.array(tokens)
    # so because gpt2 tokenizer have 50256 tokens, we can downcasting it to uint16, which is enough because 2^16 = 65536
    # this is just to save some memory and disk space
    assert (tokens_np >= 0).all() and (tokens_np < 2**16).all(), "token dictionary is too large for uint16" # here we checking the individual token ids
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


# tokenize all documents and write output shards.
# each of shard will have shard_size tokens (100M tokens)
# but the last shard may have less than shard_size tokens (has the remainder)
nprocs = max(1, os.cpu_count() // 2) # use half of the available cores, but at least 1
print(f"Using {nprocs} processes")

with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size: # if yes, add the tokens to the current shard
            # append tokens to the current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))

        else: # if no, write the current shard to disk and start a new shard
            # write the current shard to disk
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")

            # fit the remainder to the current shard; write the rest in the next shard
            remainder = shard_size - token_count

            print(f"Writing {filename} with {token_count} tokens (remainder: {remainder})")
            progress_bar.update(remainder)

            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)

            shard_index += 1
            progress_bar = None

            # start a new shard with the remaining tokens
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # write the last shard with any remaining tokens
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        print(f"Writing last {filename} with {token_count} tokens")
        write_datafile(filename, all_tokens_np[:token_count])






    



