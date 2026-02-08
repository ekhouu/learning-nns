"""
my main focus right now is implementing the transformer
will work on optimizations later maybe!
"""

import math

import torch
import torch.nn as nn
from einops import einsum, rearrange

"""
Transformer LM
IN: tensor of int shape (batch_size, seq_len)
OUT: seq of vectors (batch_size, seq_len, d_model)

Pre-norm Transformer Block
num_layers blocks
aggregate then tform (duh!)
IN: (batch_size, seq_len, d_model)
OUT: (batch_size, seq_len, d_model)

Output Norming
IN: final activations
OUT: dist
"""

"""
From assignment

Make sure to:
- subclass nn.Module
- call the superclass constructor
- construct and store your parameter as W for memory-ordering reasons, in nn.Parameter
- don't use nn.Linear or nn.functional.linear
"""


# Passes test_model.py::test_linear !!
class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(W, 0, std, -3 * std, 3 * std)

        self.weights = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        apply linear transform to input

        performs
        y = Wx
        or in NN terms
        y = x * W_T
        """

        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")


"""
As discussed above, the first layer of the Transformer is an embedding layer that maps integer token IDs
into a vector space of dimension d_model. We will implement a custom Embedding class that inherits from
torch.nn.Module (so you should not use nn.Embedding). The forward method should select the embedding
vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a
torch.LongTensor of token IDs with shape (batch_size, sequence_length)
"""


# Passes embedding tests!
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Make sure to:
        - subclass nn.Module
        - call the superclass constructor
        - initialize your embedding matrix as a nn.Parameter
        - store the embedding matrix with the d_model being in the final dimension
        - don't use nn.Embedding or nn.functional.embedding
        """

        super().__init__()

        W = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(W, 0.0, 1.0, -3, 3)
        self.weights = nn.Parameter(W)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.long:
            raise TypeError("expecting longs in embedding class!!!")

        return self.weights[token_ids]


"""
Pre-Norm Transformer Block

>>>>--|-------------------------|>>ADD>>|---------------------------|>>ADD>>...
      |                         |       |                           |
      | -> norm -> RoPE MHSA -> |       | -> norm -> pos-wise FF -> |

Root Mean Square Layer Normalization

RMSNorm(a_i in R^{d_m}) = g_i * a_i / RMS(a)

RMS(A) = math.sqrt( sum(i=1->d_m of a_i ^2 + e)* 1 / d_m)
"""


# Passes rmsnorm test!
class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        """
        Function should accept
        d_model : int     , hidden dim of model
        eps     : float   , eps value for numerical stability
        device  : device
        dtype   : dtype
        """

        self.d_model = d_model
        self.eps = eps
        self.weights = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        in: (batch_sise, seq_len, d_model)
        out: (batch_size, seq_len, d_model)
        """

        in_dtype = x.dtype
        x = x.to(torch.float32)

        # im stupid and spent 30 minutes trying ot figure out why x*x wasn't wroking as sq
        # but it's supposed ot be sum of squares not only squares
        sq = x.square().sum(dim=-1, keepdim=True)
        # rms = (sq * ).sqrt()
        result = x / (sq * (1 / self.d_model) + self.eps).sqrt() * self.weights

        return result.to(in_dtype)


"""
SiLU vs ReLU

ReLU(x) = max(0,x)
SiLU(x) = x * sigma(x)
        = x / (1 + e^-x)

SiLU smoother than ReLU ; ReLU abrupt

GLU(x,W_1,W_2) = sigma(W_1 x) ⊙ (W_2 x)
"reduce the vanishing gradient ... by providing a
linear path while retain non-linear capabilities"

FFN(x) = SwiGLU(x, W_1, W_2, W_3) = W_2 (SiLU(W_1 x) (.) W_3 x)
x in R^{d_m}, W_1 and W_3 in R^{d_ff * d_m), W_2 in R^{d_m * d_ff}
and d_ff = d_m * 8/3
"""

"""
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
"""


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        w1_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w3_weight: torch.Tensor,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Parameter(w1_weight)
        self.w2 = nn.Parameter(w2_weight)
        self.w3 = nn.Parameter(w3_weight)

        assert self.w1.shape == (d_ff, d_model)
        assert self.w3.shape == (d_ff, d_model)
        assert self.w2.shape == (d_model, d_ff)

    def forward(self, x: torch.Tensor):
        def _silu(y):
            return y * torch.sigmoid(y)

        a = einsum(x, self.w1, "... d, f d -> ... f")
        b = einsum(x, self.w3, "... d, f d -> ... f")

        h = _silu(a) * b
        # hamandodiica something idk the (.) sign

        return einsum(h, self.w2, "... f, d f -> ... d")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len=max_seq_len, device=device)

    def _build_cache(self, max_seq_len: int, device: torch.device | None):
        # makes tensor up to max_seq_len
        if device is None:
            device = self.inv_freq.device

        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        inv = self.inv_freq.to(device=device, dtype=torch.float32)
        freq = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))

        self.cos_cached = freq.cos()
        self.sin_cached = freq.sin()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: [..., seq_len, d_k]
        token_positions: [..., seq_len]
        """

        pos = token_positions.to(device=x.device, dtype=torch.long)
        need = int(pos.max().item()) + 1

        # if get issues potentially check if device conflicts
        if (self.cos_cached.device != x.device) or (self.cos_cached.size(0) < need):
            self._build_cache(max_seq_len=need, device=x.device)

        cos = self.cos_cached[pos].to(dtype=x.dtype)
        sin = self.sin_cached[pos].to(dtype=x.dtype)

        x_even, x_odd = x[..., 0::2], x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd

        return out
