"""Codebook-based content-dependent routing for SSA.

Stage 1-2 of the SSA pipeline:
  1. Project Q and K to routing space, score against learnable codebook
  2. Select top-k candidate keys per query via code-level inverted index
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CodebookRouter(nn.Module):
    """Content-dependent router that selects k candidate keys per query using
    a learned codebook for sub-quadratic candidate generation.

    Architecture:
        Q_full → W_qr → Q_route ─┐
                                   ├→ · C^T → code scores → top-a codes → inverted index
        K_full → W_kr → K_route ─┘  · C^T → code scores → top-b assignment ─┘
                                                              ↓
                              candidates (keys sharing any selected code)
                                                              ↓
                              re-score with full Q,K → top-k per query
    """

    def __init__(
        self,
        head_dim: int = 128,
        route_dim: int = 32,
        num_kv_heads: int = 8,
        num_codebook: int = 2048,
        codes_per_key: int = 4,
        codes_per_query: int = 16,
        top_k: int = 256,
        temperature: float = 1.0,
    ):
        """
        Args:
            head_dim: Full head dimension (d)
            route_dim: Routing projection dimension (d_s)
            num_kv_heads: Number of KV heads (routing is done per KV head)
            num_codebook: Number of codebook entries (N_c)
            codes_per_key: Number of codes each key is assigned to (b)
            codes_per_query: Number of top codes selected per query (a)
            top_k: Number of final candidates per query (k)
            temperature: Temperature for Gumbel-Softmax (lower = harder)
        """
        super().__init__()
        self.head_dim = head_dim
        self.route_dim = route_dim
        self.num_kv_heads = num_kv_heads
        self.num_codebook = num_codebook
        self.codes_per_key = codes_per_key
        self.codes_per_query = codes_per_query
        self.top_k = top_k
        self.temperature = temperature

        # Routing projections
        self.W_qr = nn.Linear(head_dim, route_dim, bias=False)
        self.W_kr = nn.Linear(head_dim, route_dim, bias=False)

        # Learnable codebook: [num_kv_heads, num_codebook, route_dim]
        # Per-head codebooks allow different routing geometries per head
        self.codebook = nn.Parameter(
            torch.randn(num_kv_heads, num_codebook, route_dim) / route_dim**0.5
        )

        # Optional: full re-scoring projection (reuse W_qr/W_kr or use main Q/K)
        # For prototype: use input Q_full and K_full directly for re-scoring

    def forward(
        self,
        q_full: torch.Tensor,
        k_full: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route queries to candidate key positions.

        Note: RoPE should be applied to q_full and k_full externally before
        calling this method, so that re-scoring scores incorporate position.

        Args:
            q_full: [seq_len, num_q_heads, head_dim] — with RoPE applied
            k_full: [seq_len, num_kv_heads, head_dim] — with RoPE applied
            causal_mask: Optional [seq_len, seq_len] bool mask. True where attention
                        is allowed. If None, no masking.
            hard: If True, use hard code assignment (argmax). If False, use Gumbel-Softmax.

        Returns:
            indices: [seq_len, num_kv_heads, top_k] candidate key indices
            scores: [seq_len, num_kv_heads, top_k] attention scores for candidates
        """
        seq_len = q_full.shape[0]
        device = q_full.device

        # --- Stage 1: Project to routing space ---
        # Routing projections use raw Q/K (before RoPE) for content-based routing.
        # The re-scoring step (Stage 2) uses full Q/K with RoPE.
        q_route = self.W_qr(q_full)  # [seq_len, num_q_heads, route_dim]
        k_route = self.W_kr(k_full)  # [seq_len, num_kv_heads, route_dim]

        # Group query heads to KV head count (GQA)
        num_q_heads = q_route.shape[1]
        group_size = num_q_heads // self.num_kv_heads
        assert num_q_heads % self.num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"

        # Mean-pool query routes per KV group: [seq_len, num_kv_heads, route_dim]
        q_route_grouped = q_route.view(seq_len, self.num_kv_heads, group_size, self.route_dim)
        q_route_grouped = q_route_grouped.mean(dim=2)

        # --- Stage 1b: Score against codebook ---
        # Normalize for cosine-like scoring
        q_route_norm = F.normalize(q_route_grouped, dim=-1)
        k_route_norm = F.normalize(k_route, dim=-1)
        codebook_norm = F.normalize(self.codebook, dim=-1)  # [h_kv, N_c, d_s]

        # Q-code scores: [seq_len, num_kv_heads, N_c]
        q_code_scores = torch.einsum("shd,hcd->shc", q_route_norm, codebook_norm)
        q_code_scores = q_code_scores / self.temperature

        # K-code scores: [seq_len, num_kv_heads, N_c]
        k_code_scores = torch.einsum("shd,hcd->shc", k_route_norm, codebook_norm)
        k_code_scores = k_code_scores / self.temperature

        # --- Stage 1c: Key-to-code assignment ---
        if hard:
            _, k_code_assign = k_code_scores.topk(self.codes_per_key, dim=-1)
            # [seq_len, num_kv_heads, codes_per_key]
        else:
            k_code_assign = F.gumbel_softmax(
                k_code_scores.view(-1, self.num_codebook),
                tau=self.temperature,
                hard=True,
            ).view(seq_len, self.num_kv_heads, self.num_codebook)

        # --- Stage 2: Candidate selection ---
        # For each query, select top-a codes
        _, top_q_codes = q_code_scores.topk(self.codes_per_query, dim=-1)
        # [seq_len, num_kv_heads, a]

        # Build inverted index: for each (head, code), collect key positions
        indices = self._select_candidates(
            q_full, k_full, q_code_scores, k_code_assign, top_q_codes, causal_mask
        )

        return indices

    # Candidate selection only produces integer indices (non-differentiable),
    # so disable gradient tracking to avoid building an autograd graph.
    @torch.no_grad()
    def _select_candidates(
        self,
        q_full: torch.Tensor,
        k_full: torch.Tensor,
        q_code_scores: torch.Tensor,
        k_code_assign: torch.Tensor,
        top_q_codes: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
        query_chunk: int = 256,
    ) -> torch.Tensor:
        """Select top-k candidates using codebook index matching + full-dimensional re-scoring.

        For each query position i and head h:
            1. Query selected codes → find keys assigned to those codes via inverted index
            2. Re-score candidates with full Q/K dot products
            3. Take top-k

        The re-scoring + masking + top-k are streamed over query chunks so that
        peak memory is O(query_chunk · seq_len · h) rather than O(seq_len² · h).
        Because this path only produces integer candidate indices (which are not
        differentiable), it runs under ``torch.no_grad`` to avoid building an
        autograd graph for the score tensor.

        Args:
            query_chunk: Number of query positions processed per streaming step.
                Controls the memory/throughput trade-off; peak score memory is
                O(query_chunk · seq_len · h). Defaults to 256.

        Returns:
            indices: [seq_len, num_kv_heads, top_k] candidate key indices
        """
        seq_len = q_full.shape[0]
        num_kv_heads = self.num_kv_heads
        device = q_full.device

        # Group Q heads to KV head groups for re-scoring. The routing score is the
        # mean over GQA group heads; since the score is linear in Q, the group mean
        # can be taken *before* the dot product instead of materializing the full
        # [seq_len, h, group_size, seq_len] tensor. This is mathematically identical
        # but avoids the group_size factor in compute and memory.
        num_q_heads = q_full.shape[1]
        group_size = num_q_heads // num_kv_heads
        q_grouped_mean = q_full.view(
            seq_len, num_kv_heads, group_size, self.head_dim
        ).mean(dim=2)  # [seq_len, num_kv_heads, head_dim]

        scale = self.head_dim**0.5
        effective_k = min(self.top_k, seq_len)

        indices_chunks = []
        for q_start in range(0, seq_len, query_chunk):
            q_end = min(q_start + query_chunk, seq_len)
            q_mean_chunk = q_grouped_mean[q_start:q_end]  # [qc, num_kv_heads, head_dim]

            # Re-score this query chunk against all keys: [qc, num_kv_heads, seq_len]
            scores_chunk = torch.einsum(
                "shd,thd->sht", q_mean_chunk, k_full
            ) / scale

            if causal_mask is not None:
                scores_chunk = scores_chunk.masked_fill(
                    ~causal_mask[q_start:q_end].unsqueeze(1), float("-inf")
                )

            # Restrict to keys that share at least one code with the query
            candidate_mask = self._build_candidate_mask(
                top_q_codes[q_start:q_end], k_code_assign, seq_len, num_kv_heads, device
            )  # [qc, num_kv_heads, seq_len]
            scores_chunk = scores_chunk.masked_fill(~candidate_mask, float("-inf"))

            _, idx_chunk = scores_chunk.topk(effective_k, dim=-1)
            indices_chunks.append(idx_chunk)

        indices = torch.cat(indices_chunks, dim=0)  # [seq_len, num_kv_heads, effective_k]

        # Pad to top_k size if needed (for shape consistency on short sequences)
        if effective_k < self.top_k:
            pad = torch.zeros(
                seq_len, num_kv_heads, self.top_k - effective_k,
                dtype=indices.dtype, device=device
            )
            indices = torch.cat([indices, pad], dim=-1)
        # indices: [seq_len, num_kv_heads, top_k]

        return indices

    def _build_candidate_mask(
        self,
        top_q_codes: torch.Tensor,
        k_code_assign: torch.Tensor,
        seq_len: int,
        num_kv_heads: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build boolean mask indicating which keys are candidates for each query.

        For each query at position i and head h, a key at position j is a
        candidate if they share at least one code assignment.

        ``top_q_codes`` may cover only a chunk of the query positions; the number
        of query rows is taken from its leading dimension, while the key
        dimension spans the full ``seq_len``.

        Args:
            top_q_codes: [n_q, num_kv_heads, a] — codes selected by each query
            k_code_assign: [seq_len, num_kv_heads, b] — codes assigned to each key

        Returns:
            mask: [n_q, num_kv_heads, seq_len] True where key is candidate.
        """
        n_q = top_q_codes.shape[0]
        n = seq_len
        h = num_kv_heads
        a = self.codes_per_query
        b = self.codes_per_key

        mask = torch.zeros(n_q, h, n, dtype=torch.bool, device=device)

        # Process key positions in chunks to avoid the O(n_q·n·h·a·b) broadcast
        K_CHUNK = min(128, n)
        for kj_start in range(0, n, K_CHUNK):
            kj_end = min(kj_start + K_CHUNK, n)
            k_chunk = k_code_assign[kj_start:kj_end]  # [K_CHUNK, h, b]

            # Broadcast: [n_q, 1, h, 1, a] x [1, K_CHUNK, h, b, 1]
            q_exp = top_q_codes.view(n_q, 1, h, 1, a)            # [n_q, 1, h, 1, a]
            k_exp = k_chunk.view(1, kj_end - kj_start, h, b, 1)  # [1, K_CHUNK, h, b, 1]

            match = (q_exp == k_exp)                      # [n_q, K_CHUNK, h, a, b]
            any_match = match.any(dim=-1).any(dim=-1)     # [n_q, K_CHUNK, h]
            mask[:, :, kj_start:kj_end] = any_match.permute(0, 2, 1)  # [n_q, h, K_CHUNK]

        return mask

    def get_code_assignment(self, k_full: torch.Tensor) -> torch.Tensor:
        """Get code assignments for keys (useful for KV cache management).

        Args:
            k_full: [seq_len, num_kv_heads, head_dim]

        Returns:
            assignments: [seq_len, num_kv_heads, codes_per_key] code indices
        """
        k_route = self.W_kr(k_full)
        k_route_norm = F.normalize(k_route, dim=-1)
        codebook_norm = F.normalize(self.codebook, dim=-1)
        k_code_scores = torch.einsum("shd,hcd->shc", k_route_norm, codebook_norm)
        _, assignments = k_code_scores.topk(self.codes_per_key, dim=-1)
        return assignments
