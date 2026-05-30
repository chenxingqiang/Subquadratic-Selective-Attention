"""SSAAttention — drop-in replacement for standard multi-head attention.

Combines CodebookRouter (Stages 1-2) with sparse exact attention (Stage 3).
Supports Grouped Query Attention (GQA) and RoPE position encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .router import CodebookRouter
from .attention import sparse_exact_attention
from .utils import build_causal_mask, apply_rope


class SSAAttention(nn.Module):
    """Sub-quadratic Selective Attention layer.

    Replaces the O(n²) dense Q·K^T in standard attention with a three-stage
    codebook-routed sparse pipeline. Interface matches standard multi-head
    attention for drop-in use in transformer blocks.

    Usage:
        ssa = SSAAttention(hidden_size=512, num_q_heads=8, num_kv_heads=2)
        output = ssa(hidden_states)  # same shape as input
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        route_dim: int = 16,
        num_codebook: int = 512,
        codes_per_key: int = 4,
        codes_per_query: int = 8,
        top_k: int = 64,
        bias: bool = False,
        causal: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.top_k = top_k
        # Whether to apply causal masking. True (default) for autoregressive /
        # decoder use; set False for bidirectional / encoder-style attention.
        self.causal = causal

        assert num_q_heads % num_kv_heads == 0

        # Standard QKV projections
        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=bias)

        # SSA router
        self.router = CodebookRouter(
            head_dim=head_dim,
            route_dim=route_dim,
            num_kv_heads=num_kv_heads,
            num_codebook=num_codebook,
            codes_per_key=codes_per_key,
            codes_per_query=codes_per_query,
            top_k=top_k,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
        use_dense: bool = False,
    ) -> torch.Tensor:
        """Forward pass with optional dense fallback.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional mask. For prototype: [batch, seq_len]
                           Attention is computed per batch element.
            position_ids: Optional position IDs for RoPE
            freqs: Optional pre-computed RoPE frequencies. If position_ids is
                  provided, freqs is indexed by position_ids.
            use_dense: If True, fall back to dense attention (for comparison/testing)

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Process each batch element independently
        outputs = []
        for b in range(batch_size):
            q_b = q[b]  # [seq_len, num_q_heads, head_dim]
            k_b = k[b]  # [seq_len, num_kv_heads, head_dim]
            v_b = v[b]  # [seq_len, num_kv_heads, head_dim]

            # Get RoPE freqs for this batch element if provided
            freqs_b = None
            if freqs is not None:
                if position_ids is not None:
                    freqs_b = freqs[position_ids[b]]
                else:
                    freqs_b = freqs[:seq_len]

            causal_mask = (
                build_causal_mask(seq_len, device=hidden_states.device)
                if self.causal
                else None
            )

            # Apply RoPE to full Q/K (for attention scoring) if freqs available
            if freqs is not None and freqs_b is not None:
                q_rope = apply_rope(q_b, freqs_b)
                k_rope = apply_rope(k_b, freqs_b)
            else:
                q_rope = q_b
                k_rope = k_b

            if use_dense:
                # Dense attention path (for comparison)
                from .attention import dense_attention
                out_b = dense_attention(q_rope, k_rope, v_b, causal_mask=causal_mask)
            else:
                # SSA path: router uses raw Q/K for content-based selection,
                # but re-scores with RoPE-encoded Q/K
                indices = self.router(q_rope, k_rope, causal_mask=causal_mask, hard=True)
                out_b = sparse_exact_attention(q_rope, k_rope, v_b, indices)

            outputs.append(out_b)

        output = torch.stack(outputs, dim=0)  # [batch, seq_len, num_q_heads, head_dim]
        output = output.reshape(batch_size, seq_len, -1)  # [batch, seq_len, hidden_size]
        output = self.o_proj(output)

        return output

    def compute_auxiliary_loss(
        self,
        hidden_states: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute auxiliary routing loss (KL divergence between routing scores
        and true dense attention distribution).

        This is used during training to prevent the router from collapsing to
        a lazy local-only policy. Should be called periodically, not every step.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            freqs: RoPE frequencies

        Returns:
            loss: scalar KL divergence loss
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V and reshape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        group_size = self.num_q_heads // self.num_kv_heads

        # Expand K for dense teacher
        k_exp = k.unsqueeze(3).expand(-1, -1, -1, group_size, -1).reshape(
            batch_size, seq_len, self.num_q_heads, self.head_dim
        )

        # Sample positions for efficiency
        sample_size = min(seq_len, 64)
        sample_indices = torch.randperm(seq_len, device=hidden_states.device)[:sample_size]

        causal_mask = build_causal_mask(seq_len, device=hidden_states.device)
        total_kl = torch.tensor(0.0, device=hidden_states.device)
        total_count = 0

        for b in range(batch_size):
            q_b = q[b]  # [seq_len, num_q_heads, head_dim]
            k_b = k[b]  # [seq_len, num_kv_heads, head_dim]

            # Teacher: full dense attention scores
            attn_logits = torch.einsum(
                "shd,thd->sht", q_b, k_exp[b]
            ) / (self.head_dim**0.5)
            attn_logits = attn_logits.masked_fill(
                ~causal_mask.unsqueeze(1), float("-inf")
            )
            teacher_weights = F.softmax(attn_logits, dim=-1)

            # Student: routing scores (codebook-approximated)
            student_logits = self._get_routing_logits(
                q_b, k_b, causal_mask
            )  # [seq_len, num_kv_heads, seq_len]

            # Expand student to match teacher heads
            student_logits = student_logits.unsqueeze(2).expand(
                -1, -1, group_size, -1
            ).reshape(seq_len, self.num_q_heads, seq_len)

            student_weights = F.softmax(student_logits, dim=-1)

            kl = F.kl_div(
                torch.log(student_weights[sample_indices] + 1e-10),
                teacher_weights[sample_indices],
                reduction="batchmean",
                log_target=False,
            )
            total_kl = total_kl + kl
            total_count += 1

        return total_kl / max(total_count, 1)

    def _get_routing_logits(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get routing logits (estimated attention scores from the router).

        This maps the codebook-based routing scores back to per-key scores
        for the student distribution in KL divergence computation.

        Args:
            q: [seq_len, num_q_heads, head_dim]
            k: [seq_len, num_kv_heads, head_dim]
            causal_mask: [seq_len, seq_len]

        Returns:
            routing_logits: [seq_len, num_kv_heads, seq_len]
        """
        seq_len = q.shape[0]
        device = q.device

        # Compute routing projections and code scores
        q_route = self.router.W_qr(q)
        k_route = self.router.W_kr(k)

        group_size = q.shape[1] // k.shape[1]
        q_route_grouped = q_route.reshape(
            seq_len, self.num_kv_heads, group_size, self.router.route_dim
        ).mean(dim=2)

        q_route_norm = F.normalize(q_route_grouped, dim=-1)
        k_route_norm = F.normalize(k_route, dim=-1)
        codebook_norm = F.normalize(self.router.codebook, dim=-1)

        q_code_scores = torch.einsum("shd,hcd->shc", q_route_norm, codebook_norm)
        k_code_scores = torch.einsum("shd,hcd->shc", k_route_norm, codebook_norm)

        # Estimate attention via codebook: Q·C · C^T·K^T ≈ Q·K^T in routing space
        routing_logits = torch.einsum("shc,thc->sht", q_code_scores, k_code_scores)
        routing_logits = routing_logits.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))

        return routing_logits


class ToyTransformerBlock(nn.Module):
    """Single transformer block using SSAAttention for integration testing."""

    def __init__(self, hidden_size: int = 512, **ssa_kwargs):
        super().__init__()
        self.attn = SSAAttention(hidden_size=hidden_size, **ssa_kwargs)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attn(x, **attn_kwargs)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x
