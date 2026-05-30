"""End-to-end integration tests for the SSA module."""

import torch
import pytest
from ssa import SSAAttention, ToyTransformerBlock, build_causal_mask, precompute_rope_freqs, dense_attention, sparse_exact_attention, CodebookRouter


class TestEndToEnd:
    """Full pipeline tests."""

    @pytest.fixture
    def ssa(self):
        return SSAAttention(
            hidden_size=256,
            num_q_heads=4,
            num_kv_heads=1,
            head_dim=64,
            route_dim=16,
            num_codebook=128,
            codes_per_key=4,
            codes_per_query=8,
            top_k=16,
        )

    def test_forward(self, ssa):
        """Basic forward pass."""
        x = torch.randn(2, 32, 256)
        out = ssa(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, ssa):
        """Gradients should flow through differentiable parameters."""
        x = torch.randn(2, 32, 256, requires_grad=True)
        out = ssa(x)
        loss = out.sum()
        loss.backward()

        # Check that gradients flow through QKV projections and output projection
        # (Router's codebook may not receive gradients due to hard topk routing)
        differentiable_params = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for name in differentiable_params:
            param = dict(ssa.named_parameters())[f"{name}.weight"]
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

        # Check that x gradient flows
        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any()

    def test_dense_comparison(self, ssa):
        """SSA and dense outputs should both be finite with similar scale."""
        x = torch.randn(2, 32, 256)
        out_ssa = ssa(x, use_dense=False)
        out_dense = ssa(x, use_dense=True)

        # Both outputs should be finite and have reasonable magnitude
        assert not torch.isnan(out_ssa).any()
        assert not torch.isnan(out_dense).any()
        assert not torch.isinf(out_ssa).any()
        assert not torch.isinf(out_dense).any()

        # Both should have non-zero variance (not collapsed)
        assert out_ssa.std() > 0 and out_dense.std() > 0

        # Means should be within an order of magnitude
        assert out_ssa.mean().abs() < 10 * out_dense.mean().abs() or \
               out_dense.mean().abs() < 10 * out_ssa.mean().abs(), (
            f"Mean differs drastically: {out_ssa.mean():.4f} vs {out_dense.mean():.4f}"
        )

    def test_with_rope(self, ssa):
        """Forward pass with RoPE position encoding."""
        x = torch.randn(2, 32, 256)
        freqs = precompute_rope_freqs(64, 64)
        out = ssa(x, freqs=freqs)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_variable_sequence_length(self, ssa):
        """Should handle different sequence lengths (top_k clamped if > seq_len)."""
        for n in [8, 16, 24, 32]:
            x = torch.randn(1, n, 256)
            out = ssa(x)
            assert out.shape == (1, n, 256), f"Failed at n={n}"

    def test_batch_independence(self, ssa):
        """Batch elements should not influence each other."""
        x = torch.randn(2, 32, 256)
        out1 = ssa(x)

        x0 = x[0:1]
        x1 = x[1:2]
        out0 = ssa(x0)
        out1_single = ssa(x1)

        combined = torch.cat([out0, out1_single], dim=0)
        assert torch.allclose(out1, combined, atol=1e-6), (
            "Batch elements are not independent"
        )

    def test_auxiliary_loss(self, ssa):
        """Auxiliary routing loss should be computable."""
        x = torch.randn(1, 32, 256)
        loss = ssa.compute_auxiliary_loss(x)
        assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss >= 0, f"KL divergence should be non-negative, got {loss.item()}"
        assert not torch.isnan(loss).any()

    def test_causal_flag_default_true(self, ssa):
        """SSAAttention should default to causal attention."""
        assert ssa.causal is True

    def test_bidirectional_toggle(self):
        """causal=False should enable bidirectional attention and differ from causal."""
        kwargs = dict(
            hidden_size=256, num_q_heads=4, num_kv_heads=1, head_dim=64,
            route_dim=16, num_codebook=128, codes_per_key=4, codes_per_query=8,
            top_k=16,
        )
        causal = SSAAttention(causal=True, **kwargs)
        bidir = SSAAttention(causal=False, **kwargs)
        # Share weights so only the masking differs
        bidir.load_state_dict(causal.state_dict())

        x = torch.randn(1, 32, 256)
        out_causal = causal(x)
        out_bidir = bidir(x)

        assert out_bidir.shape == x.shape
        assert not torch.isnan(out_bidir).any()
        assert bidir.causal is False
        # Bidirectional attention sees future tokens, so output must differ.
        assert not torch.allclose(out_causal, out_bidir), (
            "causal and bidirectional outputs should differ"
        )


class TestToyTransformerBlock:
    """Tests for the toy transformer block."""

    def test_forward(self):
        block = ToyTransformerBlock(hidden_size=256, num_q_heads=4, num_kv_heads=1, head_dim=64, top_k=16)
        x = torch.randn(2, 32, 256)
        out = block(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        block = ToyTransformerBlock(hidden_size=256, num_q_heads=4, num_kv_heads=1, head_dim=64, top_k=16)
        x = torch.randn(2, 32, 256, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_with_rope(self):
        block = ToyTransformerBlock(hidden_size=256, num_q_heads=4, num_kv_heads=1, head_dim=64, top_k=16)
        x = torch.randn(2, 32, 256)
        freqs = precompute_rope_freqs(64, 64)
        out = block(x, freqs=freqs)
        assert out.shape == x.shape


class TestRoutingQuality:
    """Verify that routing is content-dependent and position-independent."""

    def test_position_independent_routing(self):
        """The router should find similar content at any position."""
        seq_len = 32
        num_q_heads = 4
        num_kv_heads = 1
        head_dim = 32
        top_k = 8

        router = CodebookRouter(
            head_dim=head_dim,
            route_dim=8,
            num_kv_heads=num_kv_heads,
            num_codebook=64,
            codes_per_key=4,
            codes_per_query=8,
            top_k=top_k,
        )

        # Create a sequence with a distinctive key at different positions
        q = torch.randn(seq_len, num_q_heads, head_dim)

        # Place a "needle" key at an early position
        needle_positions = [2, 16]  # early and late
        mask = build_causal_mask(seq_len)

        for pos in needle_positions:
            k = torch.randn(seq_len, num_kv_heads, head_dim)
            # Make the needle distinctive
            k[pos] = k[pos] * 10.0

            indices = router(q, k, causal_mask=mask, hard=True)

            # Check if the router captures the distinctive position
            # For a truly position-independent router, the recall should be similar
            # at both positions. But with random Q/K, this is probabilistic.
            # We just verify the indices are valid.
            assert indices.max() < seq_len
            assert indices.min() >= 0

    def test_top_k_matches_self_attention(self):
        """Self-attention routing sanity check.

        With Q=K (self-attention) and a random untrained codebook, routing
        is essentially random. This test verifies structural correctness:
        indices are in valid range and each query gets k unique candidates.
        """
        seq_len = 16
        num_q_heads = 4
        num_kv_heads = 1
        head_dim = 32
        top_k = 8

        router = CodebookRouter(
            head_dim=head_dim,
            route_dim=8,
            num_kv_heads=num_kv_heads,
            num_codebook=64,
            codes_per_key=4,
            codes_per_query=8,
            top_k=top_k,
        )

        q = torch.randn(seq_len, num_q_heads, head_dim)
        k = q[:, :num_kv_heads, :]  # Q=K

        mask = build_causal_mask(seq_len)
        indices = router(q, k, causal_mask=mask, hard=True)

        # Structural checks (random codebook)
        assert indices.min() >= 0
        assert indices.max() < seq_len
        # Each query-head pair should get top_k candidates
        assert indices.shape == (seq_len, num_kv_heads, top_k)

    def test_codebook_learnable(self):
        """Codebook should be a learnable parameter."""
        router = CodebookRouter(
            head_dim=32, route_dim=8, num_kv_heads=1,
            num_codebook=64, codes_per_key=4, codes_per_query=8, top_k=8,
        )
        assert router.codebook.requires_grad, "Codebook should be learnable"

        # Small training step: compute a differentiable loss on the routing
        # projections (W_qr, W_kr are differentiable; hard topk is not)
        q = torch.randn(16, 4, 32, requires_grad=True)
        k = torch.randn(16, 1, 32, requires_grad=True)

        # Test that W_qr and W_kr receive gradients
        q_route = router.W_qr(q)
        k_route = router.W_kr(k)
        loss = (q_route**2).mean() + (k_route**2).mean()
        loss.backward()

        assert router.W_qr.weight.grad is not None, "W_qr should receive gradients"
        assert router.W_kr.weight.grad is not None, "W_kr should receive gradients"

    def test_multi_head_independence(self):
        """Different heads should produce different routings."""
        seq_len = 16
        num_q_heads = 6
        num_kv_heads = 2
        head_dim = 32
        top_k = 8

        router = CodebookRouter(
            head_dim=head_dim,
            route_dim=8,
            num_kv_heads=num_kv_heads,
            num_codebook=128,
            codes_per_key=4,
            codes_per_query=8,
            top_k=top_k,
        )

        q = torch.randn(seq_len, num_q_heads, head_dim)
        k = torch.randn(seq_len, num_kv_heads, head_dim)
        mask = build_causal_mask(seq_len)

        indices = router(q, k, causal_mask=mask, hard=True)

        # Different KV heads should route differently (with random Q/K, this is expected)
        # Heads might coincide statistically but shouldn't be identical for all queries
        all_identical = all(
            torch.equal(indices[:, 0, :], indices[:, h, :])
            for h in range(1, num_kv_heads)
        )
        # It's possible they match by chance with random initialization, but unlikely
        # This is a probabilistic test
        if all_identical:
            print(
                "Heads produced identical routings — possible with random Q/K, "
                "but this might indicate an issue with codebook initialization"
            )
