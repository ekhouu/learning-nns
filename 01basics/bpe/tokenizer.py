import heapq
from collections.abc import Iterable, Iterator
from functools import lru_cache
from os import PathLike

import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        cache_size: int = 32768,
    ):
        PAT = r"(?:'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"
        self.vocab = vocab
        self.special_tokens = special_tokens or []
        self.cache_size = max(0, cache_size)

        # might not need both tbh
        self.id_to_bytes = vocab
        self.bytes_to_id = {k: v for v, k in vocab.items()}

        self.merge_rank_ids = {}
        self.pair_conv = {}

        # check if enum adds overhead? idk much about py
        # it prolly doesn't add much but nice to learn
        for rank, (a, b) in enumerate(merges):
            a_id = self.bytes_to_id[a]
            b_id = self.bytes_to_id[b]
            bmerge = a + b
            nn = self.bytes_to_id[bmerge]
            self.merge_rank_ids[(a_id, b_id)] = rank
            self.pair_conv[(a_id, b_id)] = nn

        self.special_token_to_id = {}
        for s in self.special_tokens:
            sid = self.bytes_to_id[s.encode("utf-8")]
            self.special_token_to_id[s] = sid

        if self.special_tokens:
            special_pat = "|".join(
                re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)
            )
            self.special_split_re = re.compile(f"({special_pat})")
        else:
            self.special_split_re = None

        self.pretok_re = re.compile(PAT)

        self._encode_pretoken_uncached = self._encode_pretokens
        if self.cache_size > 0:
            self._encode_pretoken_cached = lru_cache(maxsize=self.cache_size)(
                self._encode_pretoken_entry
            )
        else:
            self._encode_pretoken_cached = self._encode_pretoken_entry

    def _encode_pretoken_entry(self, pretoken_bytes: bytes):
        if self._encode_pretoken_uncached is None:
            raise RuntimeError("set self._encode_pretoken_uncached first")
        out = self._encode_pretoken_uncached(pretoken_bytes)
        return out if isinstance(out, tuple) else tuple(out)

    def _encode_pretokens(
        self,
        pretoken_bytes: bytes,
    ) -> tuple[int, ...]:

        if not pretoken_bytes:
            return tuple()

        tok = [self.bytes_to_id[bytes([b])] for b in pretoken_bytes]
        n = len(tok)
        if n == 1:
            return (tok[0],)

        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 for i in range(n)]
        nxt[-1] = -1
        alive = [True] * n

        heap: list[tuple[int, int, int, int, int]] = []

        def push_pair(i: int, j: int) -> None:
            if i == -1 or j == -1:
                return
            pair = (tok[i], tok[j])

            rank = self.merge_rank_ids.get(pair)
            # DO NOT DO if rank
            # bc rank can be 0
            if rank is None:
                return

            heapq.heappush(heap, (rank, i, j, pair[0], pair[1]))

        for i in range(n - 1):
            push_pair(i, i + 1)

        while heap:
            _, i, j, a_id, b_id = heapq.heappop(heap)

            if not (alive[i] and alive[j]):
                continue
            if nxt[i] != j or prev[j] != i:
                continue
            if tok[i] != a_id or tok[j] != b_id:
                continue

            merged_id = self.pair_conv[(a_id, b_id)]

            tok[i] = merged_id
            alive[j] = False
            r = nxt[j]
            nxt[i] = r
            if r != -1:
                prev[r] = i

            l = prev[i]
            push_pair(l, i)
            push_pair(i, r)

        out: list[int] = []
        cur = 0

        while cur != -1:
            if alive[cur]:
                out.append(tok[cur])
            cur = nxt[cur]

        return tuple(out)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike,
        merges_filepath: str | PathLike,
        special_tokens: list[str] | None = None,
        cache_size: int = 32768,
    ):
        # TODO: figure out if doing this here actually helped w/ module loading?
        import json

        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_payload = json.load(f)
        with open(merges_filepath, encoding="utf-8") as f:
            merges_payload = json.load(f)

        if vocab_payload.get("format") != "cs336_bpe_vocab_hex_v1":
            raise ValueError("ts not our vocab format")
        if merges_payload.get("format") != "cs336_bpe_merges_hex_v1":
            raise ValueError("ts not merges vocab format")

        vocab_hex = vocab_payload["vocab_hex"]
        merges_hex = merges_payload["merges_hex"]

        vocab = {i: bytes.fromhex(h) for i, h in enumerate(vocab_hex)}
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_hex]

        return cls(vocab, merges, special_tokens=special_tokens, cache_size=cache_size)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        parts = self.special_split_re.split(text) if self.special_split_re else [text]

        for part in parts:
            if not part:
                continue
            sid = self.special_token_to_id.get(part)
            if sid is not None:
                ids.append(sid)
                continue
            for m in self.pretok_re.finditer(part):
                b = m.group().encode("utf-8")
                if not b:
                    continue
                ids.extend(self._encode_pretoken_cached(b))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            if not chunk:
                continue
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        # if not valid, use U+FFFD
        rep = b"\xef\xbf\xbd"
        raw = b"".join(self.id_to_bytes.get(i, rep) for i in ids)
        return raw.decode("utf-8", errors="replace")
