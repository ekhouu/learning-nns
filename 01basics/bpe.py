"""
og bpe merge implementation
"""

from array import array
from typing import Any, BinaryIO
import heapq
import multiprocessing as mp
import regex as re
import os

# TODO: list by priority
# - streaming
# - reorganize it's so ugly bro

SHIFT = 16
MASK = (1 << SHIFT) - 1

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

def task_enqueue(task_info):
    idx, chunk, special_tokens = task_info
    out = pretoken_worker(chunk, special_tokens)
    return idx, out

# it doesn't really matter what order we assemble the dataset in as long as
# we are clear about not merging through pretokens (which we are!)
def pretoken_worker(chunk: str, special_tokens):
    # the way im going to handle local n_t is prolly just adjust by the len of tokens
    # before it at merge time (which will mean we have to O(n) loop thru the arrays
    # while creating the final ver but i think it will be fast enough + multiproc
    # is a worthwhile tradeoff

    _allo = len(chunk)
    _occ = {}
    _cur = {}
    _tokens = []

    _prev = list(range(-1, _allo - 1))
    _post = list(range(1, _allo + 1))

    local_n_t = 0

    if special_tokens:
        parts = re.split(f"({special_try(special_tokens)})", chunk)
    else:
        parts = [chunk]

    for part in parts:
        if not part or part in special_tokens:
            continue
        for it in CRE.finditer(part):
            tok = [int(b) for b in bytes(it.group(), "utf-8")]
            n = len(tok)
            if n == 0:
                continue

            _tokens.extend(tok)

            for i in range(1, n):
                pk = pack(tok[i - 1], tok[i])
                if pk not in _occ:
                    _occ[pk] = 1
                    _cur[pk] = [local_n_t + i]
                else:
                    _occ[pk] += 1
                    _cur[pk].append(local_n_t + i)

            local_n_t += n

            if local_n_t+1 > _allo:
                old_allo = _allo
                _allo *= 2
                _prev.extend(range(old_allo-1, _allo -1))
                _post.extend(range(old_allo + 1, _allo + 1))

            # [h e l l o]
            # prev[5-5] = prev[0] = -1
            # post[5-1] = prev[4] = -1

            _prev[local_n_t - n] = -1
            _post[local_n_t - 1] = -1

    return {
        "local_n_t": local_n_t,
        "tokens": array('I',_tokens),
        "occ": _occ,
        "cur": {k: array('I', v) for k, v in _cur.items()},
        "prev": array('i',_prev[:local_n_t]),
        "post": array('i',_post[:local_n_t])
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
def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str],workers=4) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    n_curr = 256

    working_dict = {str(bytes(i)): i for i in range(n_curr)}

    for special in special_tokens:
        working_dict[special] = n_curr
        n_curr += 1

    fmerge = n_curr

    def _pair_key(k: int) -> tuple[_RevBytes, _RevBytes]:
        a_id, b_id = unpack(k)
        return (_RevBytes(tok_bytes[a_id]), _RevBytes(tok_bytes[b_id]))

    """
    if special_tokens:
        parts = re.split(f"({special_try})", corpus)
    else:
        parts = [corpus]
    """

    tok_bytes: list[bytes] = [b""] * n_curr
    for i in range(256):
        tok_bytes[i] = bytes([i])
    for j, s in enumerate(special_tokens):
        tok_bytes[256 + j] = s.encode("utf-8")

    # STARTING TO IMPLEMENT MULTIPROCESSING
    with open(input_path, "rb") as f:
        a = find_chunk_boundaries(f,workers,bytes("<|endoftext|>", 'utf-8'))
        # this splits into bounds
        # to get each bound
        tasks = []

        for idx, (start, end) in enumerate(zip(a[:-1], a[1:])):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            tasks.append((idx, chunk, special_tokens))

    results: list[dict[str, Any] | None] = [None] * len(tasks)

    with mp.Pool(processes=workers) as pool:
        for idx, out in pool.imap_unordered(task_enqueue, tasks, chunksize=1):
            results[idx] = out


    print("{DEBUG} All pools done!")

    n_t = 0

    occ = {}
    cur = {}

    tokens = []
    prev = []
    post = []

    # assemble actual arrays
    for result in results:
        if not result: continue

        pv = result["prev"]
        po = result["post"]

        for i in range(len(pv)):
            p = pv[i]
            if p >= 0:
                pv[i] = p + n_t

            q = po[i]
            if q >= 0:
                po[i] = q + n_t

        prev.extend(pv)
        post.extend(po)
        tokens.extend(result["tokens"])

        # for later maybe modularize
        eot_id = working_dict["<|endoftext|>"]
        tokens.append(eot_id)
        prev.append(-1)
        post.append(-1)

        for k, cnt in result["occ"].items():
            occ[k] = occ.get(k,0) + cnt

        # occurence places have to be remapped as well
        for arr in result["cur"].values():
            for j in range(len(arr)):
                arr[j] += n_t

        for a, b in result["cur"].items():
            if a in cur:
                cur[a].extend(b)
            else:
                cur[a] = b

        n_t += result["local_n_t"] + 1

    print("{DEBUG} All pools consolidated!")

    heap = [(-cnt, *_pair_key(k), k) for k, cnt in occ.items() if cnt > 0]
    heapq.heapify(heap)

    def next_key():
        while heap:
            neg_cnt, _, _, k = heap[0]
            cnt = -neg_cnt
            if occ.get(k, 0) != cnt:
                heapq.heappop(heap)
                continue
            return k
        return None

    # merge logic
    # a = prev[cr]
    # b = cr
    # l = prev[a]
    # r = post[b]
    # DISAPPEARANCES
    # (L,A) disappears if L
    # (A,B) disappears - best
    # (B,R) disappears if R

    merges = []

    while n_curr < vocab_size:
        # moved removing logic to function
        best = next_key()
        if best is None:
            break

        a_id, b_id = unpack(best)
        working_dict[best] = n_curr
        tok_bytes.append(tok_bytes[a_id] + tok_bytes[b_id])
        touched = set()

        # rewrite
        # if l is valid
        #   - create l key using tokens[prev[cr]]
        #   - occ[k] = occ.get(k,0) + 1
        #   - cur.setdefault(k, []).append(tokens[cr])

        # ok we spamming comments bc
        # im tired of looking at this and hating myself

        # FOR EVERY OCCURENCE
        for b in cur.get(best, []):
            # PAIR IS (A,B)
            # WE STORE RIGHT SIDE (B)
            # SO WE NEED TO DERIVE A
            a = prev[b]

            # ALL the cases
            # if a==-1 --> a is the start of a word
            # if a==-2 --> b has already been touched
            if a < 0:
                continue
            # if  ==-2 --> a has already been touched
            if prev[a] == -2:
                continue
            # if  !=b  --> a was touched and b was ejected
            if post[a] != b:
                continue
            # this one is basically if it has changed since we last saw
            # not sure if this is possible anymore? should whiteboard
            if tokens[a] != a_id or tokens[b] != b_id:
                continue

            l = prev[a]
            r = post[b]

            if l >= 0:
                k = pack(tokens[l], tokens[a])
                occ[k] = occ.get(k, 0) - 1
                touched.add(k)

            occ[best] = occ.get(best, 0) - 1
            touched.add(best)

            if r >= 0:
                k = pack(tokens[b], tokens[r])
                occ[k] = occ.get(k, 0) - 1
                touched.add(k)

            tokens[b] = n_curr
            # tbh i think i could only use prev but im scared to change it
            # bc it works now
            prev[a] = -2
            post[a] = -2

            prev[b] = l if l >= 0 else -1
            if l >= 0:
                post[l] = b

            post[b] = r if r >= 0 else -1
            if r >= 0:
                prev[r] = b

            # HAS TO BE DONE AFTER EVERYTHING ELSE
            if l >= 0:
                k = pack(tokens[l], tokens[b])  # since we already changed [b]
                occ[k] = occ.get(k, 0) + 1
                cur.setdefault(k, []).append(b)
                touched.add(k)

            if r >= 0:
                k = pack(tokens[b], tokens[r])
                occ[k] = occ.get(k, 0) + 1
                cur.setdefault(k, []).append(r)
                touched.add(k)

        for k in touched:
            c = occ.get(k, 0)
            if c <= 0:
                occ.pop(k, None)
            else:
                heapq.heappush(heap, (-c, *_pair_key(k), k))

        merges.append(best)
        n_curr += 1

    # DEBUG
    # thinking about merge structure
    # we start from l side of merges array
    # [12,25,39]...etc
    # we unpack
    # a_id, b_id = unpack(m)
    # so example
    # a_id, b_id = 1,2
    # then we want to convert to bytes
    # we can do it using the previous merges AND build dict at the same time

    # check if a_id and b_id are in basic bytes dict (0-255)
    # if yes, use bytes for remapping
    # if not,
    #       recursively unpack a_id and b_id until they are
    #           add to dict so we dont have to do it again

    # ok no recursion :(
    """
    with open("debug_merge.txt", "w", encoding="utf-8") as f:
        final_dict = {i: bytes([i]) for i in range(0, 256)}

        # LOW PRIORITY todo write into output (cuz memory overhead)
        def bk_unpack(token_id: int) -> bytes:
            if token_id in final_dict:
                return final_dict[token_id]

            packed = merges[token_id - fmerge]
            a_id, b_id = unpack(packed)
            a, b = bk_unpack(a_id), bk_unpack(b_id)
            final_dict[token_id] = a + b
            return final_dict[token_id]

        for m in merges:
            a_id, b_id = unpack(m)
            out_merges.append((bk_unpack(a_id), bk_unpack(b_id)))
            # f.write(f"a_id: {a_id}, b_id: {b_id} --> {out_merges[-1]}\n")
    """

    tok_bytes: list[bytes] = [b""] * n_curr
    for i in range(256):
        tok_bytes[i] = bytes([i])
    for j, s in enumerate(special_tokens):
        tok_bytes[256+j] = s.encode('utf-8')
    for i, pk in enumerate(merges):
        a_id, b_id = unpack(pk)
        new_id = fmerge + i
        tok_bytes[new_id] = tok_bytes[a_id] + tok_bytes[b_id]

    final_dict = {i : tok_bytes[i] for i in range(n_curr)}
    out_merges = [(tok_bytes[a], tok_bytes[b]) for (a, b) in (unpack(pk) for pk in merges)]

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
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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


if __name__ == "__main__":
    out_dict, out_merges = train_bpe("TinyStoriesV2-GPT4-valid.txt",10000,["<|endoftext|>"],workers=8)

    import json

    def _bytes_to_hex(b: bytes) -> str:
        return b.hex()

    def _bytes_to_str(b: bytes) -> str:
        return b.decode("utf-8", errors="backslashreplace")

    out_dict_json = {str(k): _bytes_to_hex(v) for k, v in out_dict.items()}
    out_merges_json = [[_bytes_to_str(a), _bytes_to_str(b)] for a, b in out_merges]

    with open("out_dict.json", "w", encoding="utf-8") as json_file:
        json.dump({"vocab": out_dict_json, "merges": out_merges_json}, json_file, indent=2)
