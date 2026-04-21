"""Tests for the parallel selective scan in MambaBlock.

The Heinsen-style parallel scan must be numerically equivalent to the
sequential loop scan for the same inputs, and should be meaningfully faster
for typical shapes.
"""

import time

import pytest
import torch

from tsgen.models.mamba import MambaBlock


def _clone_block_weights(src: MambaBlock, dst: MambaBlock):
    """Copy weights src -> dst so both blocks compute the same function."""
    dst.load_state_dict(src.state_dict())


class TestParallelScanEquivalence:
    """The parallel scan must match the sequential loop on the same inputs."""

    def test_matches_sequential_small(self):
        torch.manual_seed(0)
        block = MambaBlock(d_model=16, d_state=8, d_conv=4, expand=2)
        x = torch.randn(2, 12, block.d_inner)

        out_seq = block._ssm_sequential(x)
        out_par = block._ssm_parallel(x)

        torch.testing.assert_close(out_seq, out_par, rtol=1e-4, atol=1e-5)

    def test_matches_sequential_typical(self):
        """Typical Mamba shape: larger d_inner, longer seq, multiple batches."""
        torch.manual_seed(1)
        block = MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2)
        x = torch.randn(4, 64, block.d_inner)

        out_seq = block._ssm_sequential(x)
        out_par = block._ssm_parallel(x)

        torch.testing.assert_close(out_seq, out_par, rtol=1e-4, atol=1e-5)

    def test_matches_sequential_single_step(self):
        """Edge case: seq_len=1 (just one recurrence step)."""
        torch.manual_seed(2)
        block = MambaBlock(d_model=16, d_state=8, d_conv=4, expand=2)
        x = torch.randn(2, 1, block.d_inner)

        out_seq = block._ssm_sequential(x)
        out_par = block._ssm_parallel(x)

        torch.testing.assert_close(out_seq, out_par, rtol=1e-5, atol=1e-6)

    def test_matches_sequential_long(self):
        """Longer sequence (L=128): numerical stability under more accumulation."""
        torch.manual_seed(3)
        block = MambaBlock(d_model=32, d_state=16, d_conv=4, expand=2)
        x = torch.randn(2, 128, block.d_inner)

        out_seq = block._ssm_sequential(x)
        out_par = block._ssm_parallel(x)

        # Looser atol for longer sequences — accumulation amplifies fp error
        torch.testing.assert_close(out_seq, out_par, rtol=1e-3, atol=1e-4)


class TestSSMDispatch:
    """The public ssm() method should use the parallel path by default."""

    def test_ssm_matches_parallel_scan(self):
        torch.manual_seed(4)
        block = MambaBlock(d_model=16, d_state=8, d_conv=4, expand=2)
        x = torch.randn(2, 12, block.d_inner)

        out = block.ssm(x)
        out_par = block._ssm_parallel(x)

        torch.testing.assert_close(out, out_par, rtol=1e-5, atol=1e-6)


class TestSpeedup:
    """The parallel path should be faster than the sequential loop on typical shapes."""

    def test_parallel_is_faster_than_sequential(self):
        torch.manual_seed(5)
        block = MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2)
        x = torch.randn(4, 64, block.d_inner)

        # Warm up both paths (avoid first-call overhead)
        block._ssm_sequential(x)
        block._ssm_parallel(x)

        n_reps = 5
        start = time.perf_counter()
        for _ in range(n_reps):
            _ = block._ssm_sequential(x)
        t_seq = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(n_reps):
            _ = block._ssm_parallel(x)
        t_par = time.perf_counter() - start

        # On a reasonable CPU, parallel should be at least 2x faster.
        # Use a conservative threshold so the test isn't flaky.
        assert t_par < t_seq * 0.75, (
            f"Parallel path not fast enough: t_seq={t_seq:.3f}s, t_par={t_par:.3f}s"
        )


class TestFullModelUnchanged:
    """Swapping the scan implementation must not change model outputs at the
    MambaDiffusion level (guarding against accidental regressions)."""

    def test_mamba_diffusion_forward_matches_sequential(self):
        from tsgen.models.mamba import MambaDiffusion

        torch.manual_seed(6)
        model = MambaDiffusion(
            sequence_length=16, features=3, dim=32, depth=2, d_state=8,
        )
        model.eval()

        x = torch.randn(2, 16, 3)
        t = torch.randint(0, 1000, (2,))

        # Force sequential path on every MambaBlock
        with _force_sequential_scan(model):
            out_seq = model(x, t)

        out_par = model(x, t)  # default path (parallel)

        torch.testing.assert_close(out_seq, out_par, rtol=1e-3, atol=1e-4)


class _force_sequential_scan:
    """Context manager that monkey-patches MambaBlock.ssm to use the sequential scan."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        from tsgen.models.mamba import MambaBlock

        self._original = MambaBlock.ssm
        MambaBlock.ssm = MambaBlock._ssm_sequential
        return self

    def __exit__(self, *args):
        from tsgen.models.mamba import MambaBlock

        MambaBlock.ssm = self._original
