"""
og bpe merge implementation
"""

import heapq

import regex as re

# TODO: list by priority
# - output format
# - streaming


# some observations
# - special tokens never get merged cuz we only merge thru pretoks
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    corpus: str

    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    fmerge = 256
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
        if n == 0:
            continue

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

            # DO NOT MOVE THIS SECTION
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
                    heapq.heappush(heap, (-c, -k))

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

    out_merges = []

    with open("debug_merge.txt", "w", encoding="utf-8") as f:
        final_dict = {i: bytes([i]) for i in range(0, 256)}

        # TODO: write into output (cuz memory overhead)
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
            f.write(f"a_id: {a_id}, b_id: {b_id} --> {out_merges[-1]}\n")

    print(working_dict)

    # OUTPUT FORMAT
    # vocab: dict[int,bytes]
    # out_merges: list[tuple[bytes,bytes]]


train_bpe("SPECIAL_TEST.txt", 500, [])
