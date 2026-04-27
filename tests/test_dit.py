"""Tests for DiT1D (adaLN-Zero Diffusion Transformer)."""

import pytest
import torch

from tsgen.config.schema import ExperimentConfig
from tsgen.models.dit import DiT1D
from tsgen.models.registry import ModelRegistry


# ---------- construction / registration ----------

class TestConstruction:
    def test_registered_under_dit(self):
        registry = ModelRegistry.list_models()
        assert 'dit' in registry
        assert registry['dit'] is DiT1D

    def test_from_config_defaults(self):
        config = ExperimentConfig(
            model_type='dit',
            data={'tickers': ['A', 'B'], 'sequence_length': 64},
        )
        m = ModelRegistry.create(config, features=2)
        assert isinstance(m, DiT1D)
        assert m.features == 2

    def test_from_config_overrides(self):
        config = ExperimentConfig(
            model_type='dit',
            data={'tickers': ['A', 'B', 'C'], 'sequence_length': 32},
            model={'dim': 96, 'depth': 3, 'heads': 4, 'num_classes': 5},
        )
        m = ModelRegistry.create(config, features=3)
        assert m.dim == 96
        assert m.num_classes == 5
        assert len(m.blocks) == 3

    def test_output_is_zero_at_init(self):
        """Final projection is zero-initialized so training starts predicting
        zero noise. Lock in this stability property."""
        torch.manual_seed(0)
        m = DiT1D(features=3, dim=32, depth=2, heads=4, mlp_ratio=4.0)
        x = torch.randn(2, 16, 3)
        t = torch.randint(0, 1000, (2,))
        with torch.no_grad():
            out = m(x, t)
        assert torch.allclose(out, torch.zeros_like(out))


# ---------- forward / shape / flexibility ----------

class TestForward:
    def _make_model(self, features=3, dim=32, depth=2, heads=4, num_classes=0):
        m = DiT1D(
            features=features, dim=dim, depth=depth, heads=heads,
            mlp_ratio=4.0, num_classes=num_classes,
        )
        # DiT1D zero-inits every ada_ln modulation MLP and the final linear so
        # the whole network is an identity map at init. For tests that probe
        # end-to-end behavior, undo the zero-init everywhere conditioning
        # flows through.
        for block in m.blocks:
            torch.nn.init.normal_(block.ada_ln[-1].weight, std=0.02)
            torch.nn.init.normal_(block.ada_ln[-1].bias, std=0.02)
        torch.nn.init.normal_(m.final_layer.ada_ln[-1].weight, std=0.02)
        torch.nn.init.normal_(m.final_layer.ada_ln[-1].bias, std=0.02)
        torch.nn.init.normal_(m.final_proj.weight, std=0.05)
        torch.nn.init.normal_(m.final_proj.bias, std=0.05)
        return m

    def test_output_shape_equals_input_shape(self):
        m = self._make_model()
        x = torch.randn(5, 64, 3)
        t = torch.randint(0, 1000, (5,))
        out = m(x, t)
        assert out.shape == x.shape

    def test_variable_sequence_length(self):
        """Unlike DiffusionTransformer (learned positional embedding), DiT1D
        must accept any positive sequence length."""
        m = self._make_model()
        for L in [17, 41, 73, 128]:
            x = torch.randn(2, L, 3)
            t = torch.randint(0, 1000, (2,))
            out = m(x, t)
            assert out.shape == (2, L, 3)

    def test_time_conditioning_changes_prediction(self):
        m = self._make_model()
        torch.manual_seed(0)
        x = torch.randn(2, 32, 3)
        out_a = m(x, torch.full((2,), 50, dtype=torch.long))
        out_b = m(x, torch.full((2,), 500, dtype=torch.long))
        assert not torch.allclose(out_a, out_b, atol=1e-4)

    def test_class_conditioning_changes_prediction(self):
        m = self._make_model(num_classes=3)
        torch.manual_seed(0)
        x = torch.randn(2, 32, 3)
        t = torch.full((2,), 100, dtype=torch.long)
        out_a = m(x, t, y=torch.tensor([0, 0]))
        out_b = m(x, t, y=torch.tensor([2, 2]))
        assert not torch.allclose(out_a, out_b, atol=1e-4)

    def test_mask_accepted_but_ignored(self):
        """mask kwarg must be accepted (API compat)."""
        m = self._make_model()
        x = torch.randn(2, 32, 3)
        t = torch.randint(0, 1000, (2,))
        mask = torch.ones_like(x)
        out = m(x, t, mask=mask)
        assert out.shape == x.shape


# ---------- sampling integration ----------

class TestSampling:
    def test_ddpm_sampling(self):
        m = DiT1D(features=2, dim=32, depth=2, heads=4, mlp_ratio=4.0)
        m._timesteps = 50
        m._sampling_method = 'ddpm'
        out = m.generate(n_samples=2, seq_len=16, device='cpu')
        assert out.shape == (2, 16, 2)
        assert torch.isfinite(out).all()

    def test_ddim_sampling(self):
        m = DiT1D(features=2, dim=32, depth=2, heads=4, mlp_ratio=4.0)
        m._timesteps = 50
        m._sampling_method = 'ddim'
        m._num_inference_steps = 5
        out = m.generate(n_samples=2, seq_len=16, device='cpu')
        assert out.shape == (2, 16, 2)


# ---------- param count sanity ----------

class TestParameterCount:
    def test_deeper_model_has_more_params(self):
        m_shallow = DiT1D(features=2, dim=32, depth=2, heads=4, mlp_ratio=4.0)
        m_deep = DiT1D(features=2, dim=32, depth=6, heads=4, mlp_ratio=4.0)
        p_shallow = sum(p.numel() for p in m_shallow.parameters())
        p_deep = sum(p.numel() for p in m_deep.parameters())
        assert p_deep > p_shallow
