"""
og bpe merge implementation

Prolly in the future I'll rewrite with chunking

ACCREDITATION
I wrote basically everything here by hand, however I had Codex 5.3 aid me with pybind11 (see bpe_low.cpp for info).
There is a longer version of this file (well, not "longer" code-wise, but longer as in it has my early code + notes)
in the cs336_basics/OLD folder. It shows a lot of the logic I ended up moving to bpe_low.cpp.
"""

import multiprocessing as mp
import os
from array import array
from typing import Any, BinaryIO
import json

import regex as re

try:
    from . import bpe_low
except ImportError:
    try:
        import bpe_low
    except ImportError:
        import sys
        from pathlib import Path

        repo_root = str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import cs336_basics.bpe_low as bpe_low

# TODO: list by priority
# - REORGANIZE!

"""
Optimization Log
== FIRST OPTIMIZATION
- changed lists to array('I') (2.2B input token file)
- changed -1 and -2 from testing to U32_NONE and U32_TOUCHED to remove signedness
- changed how tok was built
- changed cur storage method + TRIED to rm stale
- regex ops
== SECOND OPTIMIZATION
- gave up and moved merge logic to cpp
    PIPELINE NOW IS LIKE
        python : processes file THEN makes tokens, prev, post
        cpp    : takes tokens, prev, post THEN makes occ,cur THEN does bpe merge
                 THEN sends final elements back to python
        python : takes final elements THEN serializes to json
"""

"""
TEST RESULTS
Passes all tests in 1.09s.
"""

_WORKER_F: BinaryIO
_WORKER_SPECIAL = None
_WORKER_SPECIAL_SPLIT: Any
_WORKER_SPECIAL_IDS: dict[str, int]


def _init_worker(path, special_tokens):

    global _WORKER_F
    global _WORKER_SPECIAL
    global _WORKER_SPECIAL_SPLIT
    global _WORKER_SPECIAL_IDS

    if special_tokens:
        _WORKER_SPECIAL_IDS = {s: 256 + i for i, s in enumerate(special_tokens)}
        _WORKER_SPECIAL_SPLIT = re.compile(f"({special_try(special_tokens)})")
    else:
        _WORKER_SPECIAL_IDS = {}
        _WORKER_SPECIAL_SPLIT = None

    _WORKER_F = open(path, "rb", buffering=0)
    _WORKER_SPECIAL = special_tokens


def _task(task):
    idx, start, end = task
    _WORKER_F.seek(start)
    data = _WORKER_F.read(end - start)
    chunk = data.decode("utf-8", errors="ignore")
    return idx, pretoken_worker(chunk, _WORKER_SPECIAL)


SHIFT = 16
MASK = (1 << SHIFT) - 1
U32_BITS = array("I").itemsize * 8
U32_MAX = (1 << U32_BITS) - 1
U32_TOUCHED = U32_MAX - 1
U32_NONE = U32_MAX


def pack(a: int, b: int) -> int:
    return (a << SHIFT) | b


def unpack(k: int) -> tuple[int, int]:
    return (k >> SHIFT, k & MASK)


# hd to implement this because
# lexicographically greatest stuff
PAT = r"(?:'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"
CRE = re.compile(PAT)


def special_try(special_tokens):
    return "|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True))


BPE_VOCAB_JSON_FORMAT = "cs336_bpe_vocab_hex_v1"
BPE_MERGES_JSON_FORMAT = "cs336_bpe_merges_hex_v1"


def save_bpe_json(
    vocab_output_path: str | os.PathLike,
    merges_output_path: str | os.PathLike,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> None:
    max_id = max(vocab) if vocab else -1
    vocab_hex: list[str | None] = [None] * (max_id + 1)
    for token_id, token_bytes in vocab.items():
        vocab_hex[token_id] = token_bytes.hex()

    if any(v is None for v in vocab_hex):
        raise ValueError("vocab ids must be contiguous from 0..max_id")

    vocab_payload = {
        "format": BPE_VOCAB_JSON_FORMAT,
        "vocab_hex": vocab_hex,
    }
    merges_payload = {
        "format": BPE_MERGES_JSON_FORMAT,
        "merges_hex": [[a.hex(), b.hex()] for a, b in merges],
    }

    with open(vocab_output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, ensure_ascii=True)

    with open(merges_output_path, "w", encoding="utf-8") as f:
        json.dump(merges_payload, f, ensure_ascii=True)


def load_bpe_json(
    vocab_input_path: str | os.PathLike,
    merges_input_path: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_input_path, encoding="utf-8") as f:
        vocab_payload = json.load(f)
    with open(merges_input_path, encoding="utf-8") as f:
        merges_payload = json.load(f)

    if vocab_payload.get("format") != BPE_VOCAB_JSON_FORMAT:
        raise ValueError(f"Unsupported vocab format: {vocab_payload.get('format')!r}")
    if merges_payload.get("format") != BPE_MERGES_JSON_FORMAT:
        raise ValueError(f"Unsupported merges format: {merges_payload.get('format')!r}")

    vocab_hex = vocab_payload["vocab_hex"]
    merges_hex = merges_payload["merges_hex"]

    vocab = {i: bytes.fromhex(h) for i, h in enumerate(vocab_hex)}
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_hex]
    return vocab, merges


# it doesn't really matter what order we assemble the dataset in as long as
# we are clear about not merging through pretokens (which we are!)
def pretoken_worker(chunk: str, special_tokens):
    # the way im going to handle local n_t is prolly just adjust by the len of tokens
    # before it at merge time (which will mean we have to O(n) loop thru the arrays
    # while creating the final ver but i think it will be fast enough + multiproc
    # is a worthwhile tradeoff

    _allo = max(1, len(chunk))
    _occ = {}
    _cur: dict[int, array] = {}
    _tokens = array("H")

    _prev = array("I")
    if _allo > 0:
        _prev.append(U32_NONE)
        if _allo > 1:
            _prev.extend(range(0, _allo - 1))
    _post = array("I", range(1, _allo + 1))

    local_n_t = 0

    if _WORKER_SPECIAL_SPLIT:
        parts = _WORKER_SPECIAL_SPLIT.split(chunk)
    else:
        parts = [chunk]

    for part in parts:
        if not part or part in _WORKER_SPECIAL_IDS:
            continue
        for it in CRE.finditer(part):
            tok = it.group().encode("utf-8")
            n = len(tok)
            if n == 0:
                continue

            _tokens.extend(tok)

            """
            for i in range(1, n):
                pk = pack(tok[i - 1], tok[i])
                if pk not in _occ:
                    _occ[pk] = 1
                    _cur[pk] = array("I", [local_n_t + i])
                else:
                    _occ[pk] += 1
                    _cur[pk].append(local_n_t + i)
            """

            local_n_t += n

            while local_n_t + 1 > _allo:
                old_allo = _allo
                _allo *= 2
                _prev.extend(range(old_allo - 1, _allo - 1))
                _post.extend(range(old_allo + 1, _allo + 1))

            # [h e l l o]
            # prev[5-5] = prev[0] = U32_NONE
            # post[5-1] = prev[4] = U32_NONE

            _prev[local_n_t - n] = U32_NONE
            _post[local_n_t - 1] = U32_NONE

    return {
        "local_n_t": local_n_t,
        "tokens": _tokens,
        "prev": array("I", _prev[:local_n_t]),
        "post": array("I", _post[:local_n_t]),
    }


class _RevBytes:
    __slots__ = ("b",)

    def __init__(self, b: bytes) -> None:
        self.b = b

    def __lt__(self, other: "_RevBytes") -> bool:
        return self.b > other.b

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _RevBytes):
            return False
        return self.b == other.b


# some observations
# - special tokens never get merged cuz we only merge thru pretoks
#   so we can prolly use multiprocessing to send everything between special toks
#   in to different threads
def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], workers=4
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    n_curr = 256

    working_dict = {str(bytes(i)): i for i in range(n_curr)}

    for special in special_tokens:
        working_dict[special] = n_curr
        n_curr += 1

    """
    if special_tokens:
        parts = re.split(f"({special_try})", corpus)
    else:
        parts = [corpus]
    """

    n_t = 0

    tokens = array("H")
    prev = array("I")
    post = array("I")

    def consume(result):
        nonlocal n_t, tokens, prev, post
        pv = result["prev"]
        po = result["post"]

        for i in range(len(pv)):
            p = pv[i]
            if p < U32_TOUCHED:
                pv[i] = p + n_t
            q = po[i]
            if q < U32_TOUCHED:
                po[i] = q + n_t

        prev.extend(pv)
        post.extend(po)
        tokens.extend(result["tokens"])

        """
        for k, cnt in result["occ"].items():
            occ[k] = occ.get(k, 0) + cnt

        for arr in result["cur"].values():
            for j in range(len(arr)):
                arr[j] += n_t

        for a, b in result["cur"].items():
            if a in cur:
                cur[a].extend(b)
            else:
                cur[a] = b
        """

        n_t += result["local_n_t"]

    # STARTING TO IMPLEMENT MULTIPROCESSING
    with open(input_path, "rb") as f:
        bounds = find_chunk_boundaries(f, workers * 4, bytes("<|endoftext|>", "utf-8"))
        # this splits into bounds
        # to get each bound
        tasks = ((i, s, e) for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])))

    print("{DEBUG} Launching pools...")

    with mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(input_path, special_tokens),
    ) as pool:
        pending = {}
        next_idx = 0
        for idx, result in pool.imap(_task, tasks, chunksize=1):
            pending[idx] = result
            while next_idx in pending:
                consume(pending.pop(next_idx))
                next_idx += 1

    assert n_t == len(prev) == len(tokens) == len(post)

    print("{DEBUG} All pools consolidated!")

    tok_bytes = [b""] * n_curr
    for i in range(256):
        tok_bytes[i] = bytes([i])
    for j, s in enumerate(special_tokens):
        tok_bytes[256 + j] = s.encode("utf-8")

    def _pair_key(k: int) -> tuple[_RevBytes, _RevBytes]:
        a_id, b_id = unpack(k)
        return (_RevBytes(tok_bytes[a_id]), _RevBytes(tok_bytes[b_id]))

    merges = bpe_low.bpe_merge(
        tokens=tokens,
        prev=prev,
        post=post,
        n_t=n_t,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    for packed in merges:
        a_id, b_id = unpack(packed)
        tok_bytes.append(tok_bytes[a_id] + tok_bytes[b_id])

    n_curr += len(merges)
    final_dict = {i: tok_bytes[i] for i in range(n_curr)}
    out_merges = [
        (tok_bytes[a], tok_bytes[b]) for (a, b) in (unpack(pk) for pk in merges)
    ]

    return final_dict, out_merges


"""
This is the pretokenization example provided by the CS336 staff
I've taken find_chunk_boundaries but nothing else
"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def main():
    from pathlib import Path

    input_path = Path(__file__).resolve().parents[1] / "data" / "TinyStoriesV2-GPT4-valid.txt"
    out_dict, out_merges = train_bpe(
        str(input_path), 10000, ["<|endoftext|>"], workers=8
    )

    save_bpe_json("out_vocab.json", "out_merges.json", out_dict, out_merges)

if __name__ == "__main__":
    main()