import heapq
from collections import Counter

import regex as re


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    corpus = input_path

    n_curr = 256

    working_dict = {str(bytes(i)): i for i in range(n_curr)}

    for special in special_tokens:
        working_dict[special] = n_curr
        n_curr += 1

    special_try = "|".join(
        re.escape(s) for s in sorted(special_tokens, key=len, reverse=True)
    )

    SHIFT = 16
    MASK = (1 << SHIFT) - 1

    def pack(a: int, b: int) -> int:
        return (a << SHIFT) | b

    def unpack(k: int) -> tuple[int, int]:
        return (k >> SHIFT, k & MASK)

    PAT = rf"(?:{special_try})|(?:'(?:[sdmt]|ll|ve|re)|\p{{L}}+|\p{{N}}+|[^\s\p{{L}}\p{{N}}]+)\s*|\s+"
    CRE = re.compile(PAT)

    n_t = 0

    allo = len(corpus)
    occ = {}
    cur = {}
    tokens = []

    prev = list(range(-1, allo - 1))
    post = list(range(1, allo + 1))

    for it in CRE.finditer(corpus):
        if it.group() in special_tokens:
            tokens.extend([working_dict[it.group()]])
            prev[n_t], post[n_t] = -1, -1
            n_t += 1
            continue

        tok = [int(b) for b in bytes(it.group().strip(), "utf-8")]
        n = len(tok)

        tokens.extend(tok)

        for i in range(1, n):
            pk = pack(tok[i - 1], tok[i])
            if not pk in occ:
                occ[pk] = 1
                cur[pk] = [n_t + i]
            else:
                occ[pk] += 1
                cur[pk].append(n_t + i)

        n_t += n

        if n_t > allo:
            allo *= 2
            prev.extend(i for i in range(n_t - 1, allo + 1))
            post.extend(i for i in range(n_t + 1, allo + 1))

        # [h e l l o]
        # prev[5-5] = prev[0] = -1
        # post[5-1] = prev[4] = -1

        prev[n_t - n] = -1
        post[n_t - 1] = -1

    prev = prev[:n_t]
    post = post[:n_t]

    heap = [(-cnt, -k) for k, cnt in occ.items() if cnt > 0]
    heapq.heapify(heap)

    def next_key():
        while heap:
            neg_cnt, neg_k = heap[0]
            cnt, k = -neg_cnt, -neg_k
            if occ.get(k, 0) != cnt:
                heapq.heappop(heap)
                continue
            return k
        return None

    # TODO: finish this logic
    # UPDATE LOCATIONS? maybe later
    while n_curr < vocab_size:
        best = next_key()
        if not best:
            break

        working_dict[best] = n_curr
        new_adds = Counter()

        for cr in cur[best]:
            if prev[cr] == -2 or prev[prev[cr]] == -2:
                continue

            tokens[cr] = n_curr

            # makes sure this no longer is considered
            # do when we no longer need it
            prev[cr] = prev[prev[cr]]
            post[prev[cr]] = cr

            pre_k = pack(tokens[prev[cr]], tokens[cr])
            new_adds[pre_k] += 1

            if post[cr] == -1 or prev[post[cr]] == -2:
                continue

            prev[post[cr]] = cr

            post_k = pack(tokens[cr], tokens[post[cr]])
            new_adds[post_k] += 1

        n_curr += 1
