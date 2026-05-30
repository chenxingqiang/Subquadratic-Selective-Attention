# SSA — Subquadratic Selective Attention

SSA is a research prototype of a linearly-scaling attention mechanism. It
replaces the O(n²) dense `Q·Kᵀ` computation in standard transformer attention
with a three-stage **codebook-routed sparse pipeline**, so that the per-head
cost scales linearly with sequence length `n`.

The repository contains a pure-PyTorch reference implementation, a (work in
progress) Triton kernel, training scripts, integration glue for the
[Qwen 3.6](https://huggingface.co/Qwen) family of models, and tests covering
correctness, routing quality, and scaling behaviour.

For the full architectural specification, see [`ssa_design.md`](ssa_design.md).
The validation target and motivation are described in [`target.md`](target.md).

---

## How it works

Standard attention computes `softmax(Q·Kᵀ / √d) · V`, which costs `O(n²·d)`.
SSA approximates this in three stages:

1. **Codebook routing.** A small learnable codebook `C ∈ ℝ^{N_c × d_s}` is
   shared across heads. Queries and keys are projected into a low-dimensional
   routing space (`d_s ≪ d`) and scored against the codebook. Each key is
   assigned to its top-`b` codes; each query selects its top-`a` codes.
2. **Candidate selection.** The selected codes induce, via an inverted index,
   a candidate set of approximately `k` keys per query. Candidates are
   re-scored with the full-dimensional `Q` and `K` and pruned to exactly `k`.
3. **Sparse exact attention.** Standard scaled dot-product attention is
   computed over the `k` selected keys per query, yielding the final output.

The result is exact softmax attention over a learned, content-dependent
sparse subset of the sequence. When `k = n`, SSA reduces to dense attention
(this is checked in the tests). For `k ≪ n`, total work is roughly
`O(n · N_c · d_s + n · k · d)`, which is linear in `n`.

Grouped-query attention (GQA), RoPE, and causal masking are all supported.

---

## Repository layout

```
ssa/                  Core library
├── __init__.py         Public API
├── router.py           CodebookRouter: stages 1–2 (routing + candidate selection)
├── attention.py        sparse_exact_attention (stage 3) + dense reference
├── ssa_layer.py        SSAAttention drop-in module + ToyTransformerBlock
├── utils.py            RoPE, causal masks, GQA expansion helpers
└── triton_kernel.py    Block-sparse CUDA kernel (production path)

tests/                PyTest suite
├── test_router.py        Routing recall, top-k coverage
├── test_attention.py     SSA(k=n) ≡ dense, sparse correctness
├── test_integration.py   End-to-end SSAAttention / ToyTransformerBlock
└── test_scaling.py       FLOP / time scaling vs. n

training/             Training entry points
├── train.py            Generic training loop
├── train_toy.py        Small synthetic-task training (sanity check)
├── train_qwen.py       Fine-tuning Qwen 3.6 with SSA attention swapped in
└── data.py             Dataset utilities

benchmark/            Profiling and integration scripts
├── profile_scaling.py  Measure latency / FLOPs vs. sequence length
└── qwen_integration.py Swap SSA into a Qwen model and run inference

ssa_design.md         Full architecture specification
target.md             Project goals and validation target
AGENTS.md             Notes for AI coding agents working in this repo
```

---

## Installation

The prototype targets Python ≥ 3.10 and PyTorch ≥ 2.1.

```bash
git clone https://github.com/chenxingqiang/Subquadratic-Selective-Attention.git
cd Subquadratic-Selective-Attention

# Recommended: a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the `ssa` package (editable) plus its runtime dependency
pip install -e .

# For the test suite:
pip install -e ".[test]"
# Optional, for the production sparse path and Qwen integration (CUDA only):
pip install -e ".[prod]"
```

After `pip install -e .` the `ssa` package is importable from anywhere; you no
longer need to run from the repository root. Alternatively, install just the
dependencies with `pip install -r requirements.txt`.

---

## Quick start

Use `SSAAttention` as a drop-in replacement for scaled dot-product attention.
The constructor uses descriptive names (the short symbols from
[`ssa_design.md`](ssa_design.md) are noted in comments):

```python
import torch
from ssa import SSAAttention

n, hidden_size = 1024, 512

attn = SSAAttention(
    hidden_size=hidden_size,   # d_model
    num_q_heads=8,             # h_q  (GQA: 8 query heads)
    num_kv_heads=2,            # h_kv (GQA: 2 KV heads)
    head_dim=64,               # d_head
    route_dim=32,              # d_s — routing dimension
    num_codebook=2048,         # N_c — codebook size
    codes_per_key=4,           # b — codes per key
    codes_per_query=16,        # a — codes per query
    top_k=256,                 # k — candidates per query
    causal=True,               # set False for bidirectional / encoder use
)

x = torch.randn(2, n, hidden_size)
y = attn(x)                    # (2, n, hidden_size)
```

Or use the lower-level pieces directly:

```python
import torch
from ssa import CodebookRouter, sparse_exact_attention, build_causal_mask

n, h_q, h_kv, d_head = 64, 8, 2, 64
Q = torch.randn(n, h_q, d_head)
K = torch.randn(n, h_kv, d_head)
V = torch.randn(n, h_kv, d_head)

router = CodebookRouter(
    head_dim=d_head, route_dim=32, num_kv_heads=h_kv,
    num_codebook=2048, codes_per_key=4, codes_per_query=16, top_k=32,
)
mask = build_causal_mask(n)                          # omit for non-causal
candidates = router(Q, K, causal_mask=mask, hard=True)  # [n, h_kv, top_k] indices
out = sparse_exact_attention(Q, K, V, candidates)       # exact attention on those keys
```

---

## Testing

Run the full suite:

```bash
pytest -q
```

Notable invariants checked:

- **Correctness floor:** with `k = n`, SSA output equals dense attention
  within numerical tolerance.
- **Routing recall:** the candidate set captures ≥ 90% of the true top-`k`
  keys from full dense attention on randomized inputs.
- **Linear scaling:** FLOPs as a function of `n` fit a linear model with
  `R² > 0.99`.

---

## Training and Qwen integration

- `training/train_toy.py` runs the minimal `ToyTransformerBlock` on a
  synthetic task — useful for verifying that the routing components learn
  before scaling up.
- `training/train_qwen.py` and `benchmark/qwen_integration.py` swap SSA into
  Qwen 3.6 attention blocks. The new parameters (routing projections
  `W_qr`, `W_kr` and the codebook `C`) are initialized randomly; the rest of
  the checkpoint is loaded as-is and either fine-tuned or frozen. See
  [`ssa_design.md`](ssa_design.md) for details on weight compatibility, the
  Gumbel-softmax annealing schedule, and the auxiliary routing loss.

---

## Status

This is an **early research prototype**. The pure-PyTorch path is the
reference implementation and is what the tests exercise. The Triton sparse
kernel and the Qwen 3.6 integration are still being validated and may
change. Contributions and bug reports are welcome.

## License

No license file is present yet; treat the code as "all rights reserved" by
the repository owner until one is added.
