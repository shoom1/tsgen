"""Tests for DiffWave1D dilated-conv diffusion model."""

import math
import pytest
import torch

from tsgen.config.schema import ExperimentConfig
from tsgen.models.diffwave import DiffWave1D
from tsgen.models.registry import ModelRegistry


# ---------- construction / registration ----------

class TestConstruction:
    def test_registered_under_diffwave(self):
        registry = ModelRegistry.list_models()
        assert 'diffwave' in registry
        assert registry['diffwave'] is DiffWave1D

    def test_from_config_defaults(self):
        config = ExperimentConfig(
            model_type='diffwave',
            data={'tickers': ['A', 'B'], 'sequence_length': 64},
        )
        m = ModelRegistry.create(config, features=2)
        assert isinstance(m, DiffWave1D)
        assert m.features == 2

    def test_output_is_zero_at_init(self):
        """DiffWave zero-initializes output_proj so training starts predicting
        zero noise. This is a load-bearing stability property — lock it in."""
        torch.manual_seed(0)
        m = DiffWave1D(
            sequence_length=None, features=2,
            residual_channels=16, num_blocks=4, dilation_cycle_length=4,
        )
        x = torch.randn(3, 32, 2)
        t = torch.randint(0, 1000, (3,))
        with torch.no_grad():
            out = m(x, t)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_from_config_overrides(self):
        config = ExperimentConfig(
            model_type='diffwave',
            data={'tickers': ['A'], 'sequence_length': 32},
            model={
                'residual_channels': 32,
                'num_blocks': 6,
                'dilation_cycle_length': 3,
                'num_classes': 4,
            },
        )
        m = ModelRegistry.create(config, features=1)
        assert m.num_classes == 4
        assert len(m.blocks) == 6


# ---------- forward / shape / flexibility ----------

class TestForward:
    def _make_model(self, features=3, residual_channels=16, num_blocks=4, dilation_cycle_length=4, num_classes=0):
        m = DiffWave1D(
            sequence_length=None,  # not required by the arch
            features=features,
            residual_channels=residual_channels,
            num_blocks=num_blocks,
            dilation_cycle_length=dilation_cycle_length,
            num_classes=num_classes,
        )
        # DiffWave zero-initializes output_proj for training stability, which
        # makes *any* forward pass return zeros. Randomize for tests that
        # probe end-to-end behavior.
        torch.nn.init.normal_(m.output_proj.weight, std=0.05)
        torch.nn.init.normal_(m.output_proj.bias, std=0.05)
        return m

    def test_output_shape_equals_input_shape(self):
        m = self._make_model()
        x = torch.randn(5, 64, 3)
        t = torch.randint(0, 1000, (5,))
        out = m(x, t)
        assert out.shape == x.shape

    def test_no_sequence_length_divisibility_requirement(self):
        """Unlike UNet1D, DiffWave must accept arbitrary sequence lengths."""
        m = self._make_model()
        for L in [17, 41, 73, 128]:
            x = torch.randn(2, L, 3)
            t = torch.randint(0, 1000, (2,))
            out = m(x, t)
            assert out.shape == (2, L, 3), f"Failed for L={L}"

    def test_time_conditioning_changes_prediction(self):
        m = self._make_model()
        torch.manual_seed(0)
        x = torch.randn(2, 32, 3)
        out_a = m(x, torch.full((2,), 50, dtype=torch.long))
        out_b = m(x, torch.full((2,), 500, dtype=torch.long))
        # Two different timesteps must produce meaningfully different predictions
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
        """mask kwarg must be accepted (API compat) even if the arch ignores it."""
        m = self._make_model()
        x = torch.randn(2, 32, 3)
        t = torch.randint(0, 1000, (2,))
        mask = torch.ones_like(x)
        out = m(x, t, mask=mask)
        assert out.shape == x.shape

    def test_receptive_field_covers_window(self):
        """With dilations [1,2,4,8] (cycle_length=4) and kernel 3, the receptive
        field is 1 + 2*(1+2+4+8)*(num_blocks/cycle) ≥ 31 per cycle.
        Perturbing the middle of the input must change predictions at both ends
        of the window for a sufficiently deep stack."""
        m = self._make_model(num_blocks=8, dilation_cycle_length=4)
        m.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 32, 3)
        t = torch.full((1,), 100, dtype=torch.long)
        with torch.no_grad():
            out_base = m(x, t)

        x_perturbed = x.clone()
        x_perturbed[0, 16, :] += 5.0  # large perturbation at center
        with torch.no_grad():
            out_pert = m(x_perturbed, t)

        diff = (out_pert - out_base).abs().mean(dim=-1).squeeze(0)  # (L,)
        # Edges must see some nonzero change from a center perturbation
        assert diff[0] > 1e-6
        assert diff[-1] > 1e-6


# ---------- diffusion sampling integration ----------

class TestSampling:
    def test_generate_via_base_class(self):
        """DiffWave1D must integrate with DiffusionModel.generate() out of the box."""
        m = DiffWave1D(
            sequence_length=None, features=2,
            residual_channels=16, num_blocks=4, dilation_cycle_length=4,
        )
        # Minimal diffusion config
        m._timesteps = 50
        m._sampling_method = 'ddpm'
        out = m.generate(n_samples=2, seq_len=16, device='cpu')
        assert out.shape == (2, 16, 2)
        assert torch.isfinite(out).all()

    def test_generate_ddim(self):
        m = DiffWave1D(
            sequence_length=None, features=2,
            residual_channels=16, num_blocks=4, dilation_cycle_length=4,
        )
        m._timesteps = 50
        m._sampling_method = 'ddim'
        m._num_inference_steps = 5
        out = m.generate(n_samples=2, seq_len=16, device='cpu')
        assert out.shape == (2, 16, 2)


# ---------- param count sanity ----------

class TestParameterCount:
    def test_more_blocks_more_params(self):
        m_small = DiffWave1D(
            sequence_length=None, features=2,
            residual_channels=16, num_blocks=4, dilation_cycle_length=4,
        )
        m_large = DiffWave1D(
            sequence_length=None, features=2,
            residual_channels=16, num_blocks=8, dilation_cycle_length=4,
        )
        p_small = sum(p.numel() for p in m_small.parameters())
        p_large = sum(p.numel() for p in m_large.parameters())
        assert p_large > p_small
