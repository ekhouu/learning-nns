import numpy as np
from .tokenizer import Tokenizer

VOCAB_PATH, MERGE_PATH = "vm_outputs/out_tiny_vocab.json", "vm_outputs/out_tiny_merges.json"
SPECIAL_TOKENS = ["<|endoftext|>"]

CORPUS_PATH = "data/TinyStoriesV2-GPT4-train.txt"
OUT_PATH = "out/TOKENIZE_tinystories_train.bin"

tokizer = Tokenizer.from_files(VOCAB_PATH, MERGE_PATH, SPECIAL_TOKENS)

CHUNK = 1_000_000

buf = np.empty(CHUNK, dtype=np.uint16)

with open(CORPUS_PATH, encoding="utf-8") as fin:
    with open(OUT_PATH, "wb") as fout:
        i = 0

        for tok_id in tokizer.encode_iterable(fin):
            buf[i] = tok_id
            i += 1

            if i == CHUNK:
                buf.tofile(fout)
                i = 0

        if i != 0:
            buf.tofile(fout)
