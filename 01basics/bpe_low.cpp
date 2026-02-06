/* ACCREDITATION
 * I wrote my original BPE merge logic (see: cs336_basics/OLD/bpe.py) in Python,
 * then moved it to CPP by hand. However, I have never used pybind11 before, and
 * had Codex 5.3 help me out with that. It also aided me in troubleshooting some
 * memory issues and organizing this file better.
 * */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

constexpr uint32_t SHIFT = 16;
constexpr uint32_t MASK = (1u << SHIFT) - 1;
constexpr uint32_t U32_MAX = std::numeric_limits<uint32_t>::max();
constexpr uint32_t U32_TOUCHED = U32_MAX - 1;
constexpr uint32_t U32_NONE = U32_MAX;
constexpr uint32_t MAX_VOCAB =
    static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()) + 1u;

inline uint32_t pack(uint16_t a, uint16_t b) {
  return (uint32_t(a) << SHIFT) | uint32_t(b);
}

inline std::pair<uint16_t, uint16_t> unpack(uint32_t k) {
  return {static_cast<uint16_t>(k >> SHIFT), static_cast<uint16_t>(k & MASK)};
}

struct PairInfo {
  int64_t count = 0;
  std::vector<uint32_t> pos;
};

struct HeapItem {
  int64_t cnt;
  uint16_t a, b;
  uint32_t k;
};

inline bool lex_greater(std::string_view a, std::string_view b) {
  const size_t n = std::min(a.size(), b.size());
  const int cmp = std::memcmp(a.data(), b.data(), n);
  if (cmp != 0)
    return cmp > 0;
  return a.size() > b.size();
}

struct HeapCmp {
  const std::vector<std::string> *tok;

  bool operator()(const HeapItem &x, const HeapItem &y) const {
    if (x.cnt != y.cnt)
      return x.cnt < y.cnt;

    const auto &ax = (*tok)[x.a];
    const auto &ay = (*tok)[y.a];
    if (ax != ay)
      return !lex_greater(ax, ay);

    const auto &bx = (*tok)[x.b];
    const auto &by = (*tok)[y.b];
    if (bx != by)
      return !lex_greater(bx, by);

    return x.k > y.k;
  }
};

inline void append_occurrence(std::unordered_map<uint32_t, PairInfo> &cur,
                              uint32_t k, uint32_t right_pos) {
  cur[k].pos.push_back(right_pos);
}

void build_occ_cur(const uint16_t *tokens, const uint32_t *prev, uint32_t n_t,
                   std::unordered_map<uint32_t, PairInfo> &cur) {
  const size_t reserve_hint =
      std::min<size_t>(static_cast<size_t>(n_t / 32), 8u * 1024u * 1024u);
  if (reserve_hint > 0) {
    cur.reserve(reserve_hint);
  }

  for (uint32_t b = 0; b < n_t; b++) {
    uint32_t a = prev[b];
    if (a >= U32_TOUCHED)
      continue;

    uint16_t a_id = tokens[a];
    uint16_t b_id = tokens[b];
    uint32_t k = pack(a_id, b_id);

    PairInfo &info = cur[k];
    info.count += 1;
  }

  for (auto &[k, info] : cur) {
    static_cast<void>(k);
    info.pos.reserve(static_cast<size_t>(info.count));
  }

  for (uint32_t b = 0; b < n_t; b++) {
    uint32_t a = prev[b];
    if (a >= U32_TOUCHED)
      continue;

    uint16_t a_id = tokens[a];
    uint16_t b_id = tokens[b];
    uint32_t k = pack(a_id, b_id);
    cur[k].pos.push_back(b);
  }
}

inline std::optional<uint32_t>
next_key(std::vector<HeapItem> &heap, const HeapCmp &cmp,
         const std::unordered_map<uint32_t, PairInfo> &cur) {
  while (!heap.empty()) {
    const HeapItem &top = heap.front();
    const auto it = cur.find(top.k);
    const int64_t cur_count = (it == cur.end()) ? 0 : it->second.count;

    if (cur_count <= 0 || cur_count != top.cnt) {
      std::pop_heap(heap.begin(), heap.end(), cmp);
      heap.pop_back();
      continue;
    }
    return top.k;
  }
  return std::nullopt;
}

inline std::string special_to_bytes(const py::handle &obj) {
  if (py::isinstance<py::bytes>(obj)) {
    return obj.cast<std::string>();
  }
  if (py::isinstance<py::str>(obj)) {
    return obj.cast<std::string>();
  }
  throw std::runtime_error("special_tokens must contain str or bytes values");
}

py::list bpe_merge(py::buffer tokens_buf, py::buffer prev_buf,
                   py::buffer post_buf, uint32_t n_t, uint32_t vocab_size,
                   py::iterable special_tokens) {
  auto tokens = tokens_buf.request();
  auto prev = prev_buf.request();
  auto post = post_buf.request();

  if (tokens.ndim != 1 || prev.ndim != 1 || post.ndim != 1)
    throw std::runtime_error("buffers malformed");

  if (tokens.format != py::format_descriptor<uint16_t>::format())
    throw std::runtime_error("tokens not uint16");

  if (prev.format != py::format_descriptor<uint32_t>::format() ||
      post.format != py::format_descriptor<uint32_t>::format())
    throw std::runtime_error("prev or post not uint32");

  if (tokens.strides[0] != static_cast<py::ssize_t>(sizeof(uint16_t)) ||
      prev.strides[0] != static_cast<py::ssize_t>(sizeof(uint32_t)) ||
      post.strides[0] != static_cast<py::ssize_t>(sizeof(uint32_t)))
    throw std::runtime_error("buffers must be contiguous");

  if (tokens.readonly || prev.readonly || post.readonly)
    throw std::runtime_error("buffers must be writable");

  if (tokens.size != prev.size || tokens.size != post.size ||
      tokens.size != static_cast<py::ssize_t>(n_t))
    throw std::runtime_error("sizes not the same");

  if (vocab_size > MAX_VOCAB)
    throw std::runtime_error(
        "vocab_size exceeds uint16 token id limit (65536)");

  std::vector<std::string> special_bytes;
  for (py::handle obj : special_tokens) {
    special_bytes.push_back(special_to_bytes(obj));
  }

  const uint32_t base_vocab =
      256u + static_cast<uint32_t>(special_bytes.size());
  if (base_vocab > MAX_VOCAB)
    throw std::runtime_error(
        "too many special tokens for uint16 token id limit");

  auto *t = static_cast<uint16_t *>(tokens.ptr);
  auto *p = static_cast<uint32_t *>(prev.ptr);
  auto *q = static_cast<uint32_t *>(post.ptr);

  std::vector<uint32_t> merges;
  if (vocab_size <= base_vocab) {
    return py::list();
  }

  {
    py::gil_scoped_release release;

    std::unordered_map<uint32_t, PairInfo> cur;
    build_occ_cur(t, p, n_t, cur);
    cur.reserve(cur.size() +
                static_cast<size_t>(2) * (vocab_size - base_vocab));

    std::vector<std::string> tok_bytes;
    tok_bytes.reserve(vocab_size);
    for (uint32_t i = 0; i < 256u; ++i) {
      tok_bytes.emplace_back(1, static_cast<char>(i));
    }
    for (const std::string &token : special_bytes) {
      tok_bytes.push_back(token);
    }

    std::vector<HeapItem> heap;
    heap.reserve(cur.size());
    for (auto &[k, info] : cur) {
      if (info.count <= 0)
        continue;
      const auto [a_id, b_id] = unpack(k);
      heap.push_back({info.count, a_id, b_id, k});
    }
    HeapCmp cmp{&tok_bytes};
    std::make_heap(heap.begin(), heap.end(), cmp);

    merges.reserve(vocab_size - base_vocab);
    uint32_t n_curr = base_vocab;

    while (n_curr < vocab_size) {
      const auto best_opt = next_key(heap, cmp, cur);
      if (!best_opt.has_value()) {
        break;
      }
      const uint32_t best = *best_opt;
      const auto [a_id, b_id] = unpack(best);

      tok_bytes.push_back(tok_bytes[a_id] + tok_bytes[b_id]);
      std::unordered_set<uint32_t> touched;
      touched.reserve(8);

      auto best_it = cur.find(best);
      if (best_it == cur.end()) {
        break;
      }

      for (uint32_t b : best_it->second.pos) {
        const uint32_t a = p[b];

        if (a >= U32_TOUCHED) {
          continue;
        }
        if (p[a] == U32_TOUCHED) {
          continue;
        }
        if (q[a] != b) {
          continue;
        }
        if (t[a] != a_id || t[b] != b_id) {
          continue;
        }

        const uint32_t l = p[a];
        const uint32_t r = q[b];

        if (l < U32_TOUCHED) {
          const uint32_t k = pack(t[l], t[a]);
          cur[k].count -= 1;
          touched.insert(k);
        }

        cur[best].count -= 1;
        touched.insert(best);

        if (r < U32_TOUCHED) {
          const uint32_t k = pack(t[b], t[r]);
          cur[k].count -= 1;
          touched.insert(k);
        }

        t[b] = static_cast<uint16_t>(n_curr);
        p[a] = U32_TOUCHED;
        q[a] = U32_TOUCHED;

        p[b] = (l < U32_TOUCHED) ? l : U32_NONE;
        if (l < U32_TOUCHED) {
          q[l] = b;
        }

        q[b] = (r < U32_TOUCHED) ? r : U32_NONE;
        if (r < U32_TOUCHED) {
          p[r] = b;
        }

        if (l < U32_TOUCHED) {
          const uint32_t k = pack(t[l], t[b]);
          cur[k].count += 1;
          append_occurrence(cur, k, b);
          touched.insert(k);
        }

        if (r < U32_TOUCHED) {
          const uint32_t k = pack(t[b], t[r]);
          cur[k].count += 1;
          append_occurrence(cur, k, r);
          touched.insert(k);
        }
      }

      for (uint32_t k : touched) {
        const auto it = cur.find(k);
        const int64_t c = (it == cur.end()) ? 0 : it->second.count;

        if (c <= 0) {
          if (it != cur.end()) {
            cur.erase(it);
          }
          continue;
        }

        const auto [left_id, right_id] = unpack(k);
        heap.push_back({c, left_id, right_id, k});
        std::push_heap(heap.begin(), heap.end(), cmp);
      }

      merges.push_back(best);
      cur.erase(best);
      n_curr++;
      if ((n_curr % 100) == 0)
        std::cout << "Dict size: " << n_curr << std::endl;
    }
  }

  py::list out;
  for (uint32_t packed : merges) {
    out.append(py::int_(packed));
  }
  return out;
}

PYBIND11_MODULE(bpe_low, m) {
  m.def("bpe_merge", &bpe_merge, py::arg("tokens"), py::arg("prev"),
        py::arg("post"), py::arg("n_t"), py::arg("vocab_size"),
        py::arg("special_tokens") = py::tuple(),
        "Run the core BPE merge loop and return packed merge pairs.");
}
