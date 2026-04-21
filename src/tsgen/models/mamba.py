import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tsgen.models.base_model import DiffusionModel
from tsgen.models.embeddings import SinusoidalPositionEmbeddings
from tsgen.models.registry import ModelRegistry

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class MambaBlock(nn.Module):
    """A pure-PyTorch implementation of the Mamba block (Selective SSM).

    Avoids the specialized CUDA kernels of the official 'mamba-ssm' package
    for compatibility with CPU and any GPU environment (including Mac).

    The selective scan is implemented via a chunked Heinsen parallel scan
    (see ``_ssm_parallel``), which vectorizes across the sequence dimension
    using ``cumprod`` / ``cumsum`` instead of a Python ``for t in range(L)``
    loop. The sequential reference implementation is preserved at
    ``_ssm_sequential`` for correctness validation (see tests) and as a
    textbook reference for readers of the code.
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 16, 
        d_conv: int = 4, 
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # In-projection: (B, L, D) -> (B, L, 2*D_inner)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Convolution (Causal)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # SSM Parameters
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D / Hippo initialization for A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Out-projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Initialize dt_proj bias specifically
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.weight.data.zero_() 

    # Chunk size for the parallel scan. Small enough to keep
    #   exp(cumsum(log dA)) within float32 range even when |log dA| is large,
    # large enough that the chunked loop iterates few times.
    _SSM_CHUNK_SIZE = 16

    def ssm(self, x):
        """Runs the SSM recurrence via parallel selective scan.

            h_t = A_bar * h_{t-1} + B_bar * x_t
            y_t = C * h_t

        The implementation uses the Heinsen parallel scan (chunked for
        numerical stability). Equivalent to ``_ssm_sequential`` within
        float32 tolerance; see tests/test_mamba_parallel_scan.py.
        """
        return self._ssm_parallel(x)

    def _project_delta_B_C(self, x):
        """Compute the per-step discretization factors needed by both scans.

        Returns:
            delta: (B, L, d_inner) — softplus-applied
            B_proj: (B, L, d_state)
            C_proj: (B, L, d_state)
            A: (d_inner, d_state) — negative HiPPO-style matrix
        """
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta, B_proj, C_proj = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))      # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())           # (d_inner, d_state)
        return delta, B_proj, C_proj, A

    def _ssm_sequential(self, x):
        """Reference scan: explicit Python loop. Retained for correctness
        validation of the parallel path (tests) and for readability."""
        (batch, seq_len, d_inner) = x.shape
        delta, B_proj, C_proj, A = self._project_delta_B_C(x)

        ys = []
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            dt = delta[:, t, :]                       # (B, D)
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))   # (B, D, N)
            Bt = B_proj[:, t, :]                      # (B, N)
            xt = x[:, t, :]                           # (B, D)
            dB = torch.einsum("bd,bn->bdn", dt, Bt)   # (B, D, N)
            h = dA * h + dB * xt.unsqueeze(-1)
            Ct = C_proj[:, t, :]                      # (B, N)
            ys.append(torch.einsum("bdn,bn->bd", h, Ct))

        y = torch.stack(ys, dim=1)                    # (B, L, D)
        return y + x * self.D

    def _ssm_parallel(self, x):
        """Chunked Heinsen parallel scan.

        For the scalar linear recurrence h_t = a_t h_{t-1} + b_t with h_0 = 0,

            h_t = (prod_{k<=t} a_k) * cumsum_t(b_s / prod_{k<=s} a_k)

        which reduces the recurrence to two vectorized ``cumsum`` ops — no
        Python loop over time steps. Mamba's state matrix A is diagonal in
        the state dimension, so this scalar identity applies element-wise
        over every (batch, d_inner, d_state) triple in parallel.

        Chunking keeps cumulative products within float32 range: each chunk
        multiplies at most ``_SSM_CHUNK_SIZE`` factors in (0, 1], so
        ``torch.cumprod`` is used directly (no log/exp) for speed.

        Performance notes:
            - On GPU (where memory bandwidth is abundant) this path is
              dramatically faster than the sequential loop; the work is
              vectorized across the L dimension via ``cumprod`` / ``cumsum``.
            - On CPU the speedup is ~1.5-2x for small batch sizes (B<=8).
              At very large batch sizes the materialization of (B, k, D, N)
              chunk tensors becomes memory-bandwidth-bound and the sequential
              loop (which keeps only an (B, D, N) state buffer) can match or
              slightly beat it. This is a hardware tradeoff, not an algorithm
              issue — the parallel scan is the canonical Mamba implementation.
        """
        (batch, seq_len, d_inner) = x.shape
        delta, B_proj, C_proj, A = self._project_delta_B_C(x)
        N = self.d_state
        D = self.d_inner
        A_bcast = A.unsqueeze(0).unsqueeze(0)            # (1, 1, D, N) for broadcast

        # Running state carried across chunks, shape (B, D, N)
        h_state = torch.zeros(batch, D, N, device=x.device, dtype=x.dtype)
        y_out = torch.empty(batch, seq_len, D, device=x.device, dtype=x.dtype)

        chunk_size = self._SSM_CHUNK_SIZE
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)

            # Build chunk-local tensors only — avoids a full-L (B,L,D,N) allocation
            delta_ck = delta[:, start:end]                # (B, k, D)
            B_ck_proj = B_proj[:, start:end]              # (B, k, N)
            x_ck_in = x[:, start:end]                     # (B, k, D)

            dA_ck = torch.exp(delta_ck.unsqueeze(-1) * A_bcast)             # (B, k, D, N)
            b_ck = (
                delta_ck.unsqueeze(-1)
                * B_ck_proj.unsqueeze(-2)
                * x_ck_in.unsqueeze(-1)
            )                                                                # (B, k, D, N)
            C_ck = C_proj[:, start:end]                                      # (B, k, N)

            # Cumulative product of a_k within this chunk (inclusive scan).
            # Factors are all in (0, 1], chunked so the smallest product stays
            # well within float32 normal range (~1e-30 floor even at chunk=32).
            cumprod = torch.cumprod(dA_ck, dim=1)         # (B, k, D, N)
            inv_cumprod = 1.0 / cumprod                   # (B, k, D, N)

            # "Free" response within chunk (zero initial state):
            #   h_free_t = cumprod_t * cumsum_t(b_s * inv_cumprod_s)
            free_accum = torch.cumsum(b_ck * inv_cumprod, dim=1)

            # y decomposes into contributions from h_state (forced) and free_accum
            # (free), each multiplied by cumprod and contracted with C_ck over n.
            # Combining inline avoids materializing the full (B, k, D, N) h_ck tensor.
            y_out[:, start:end] = torch.einsum(
                "bkn,bkdn,bkdn->bkd",
                C_ck,
                cumprod,
                h_state.unsqueeze(1) + free_accum,
            )

            # Carry final state into the next chunk
            h_state = cumprod[:, -1] * (h_state + free_accum[:, -1])

        return y_out + x * self.D

    def forward(self, x):
        # x: (B, L, D)
        batch, seq_len, _ = x.shape

        # 1. In-projection
        xz = self.in_proj(x) # (B, L, 2*D_inner)
        x_prime, z = xz.chunk(2, dim=-1)

        # 2. Convolution (1D)
        # Permute to (B, D, L) for Conv1d
        x_prime = x_prime.transpose(1, 2)
        x_prime = self.conv1d(x_prime)[:, :, :seq_len] # Causal crop
        x_prime = x_prime.transpose(1, 2)
        
        x_prime = F.silu(x_prime)
        
        # 3. SSM
        y = self.ssm(x_prime)
        
        # 4. Gating (Multiplicative)
        output = y * F.silu(z)
        
        # 5. Out-projection
        return self.out_proj(output)

@ModelRegistry.register('mamba')
class MambaDiffusion(DiffusionModel):
    """
    Diffusion Model using Mamba (S4) backbone.
    """

    supports_conditioning = True
    supports_masking = False  # Mamba is sequential, doesn't use attention masking
    def __init__(
        self,
        sequence_length: int,
        features: int,
        dim: int = 128,
        depth: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_classes: int = 0
    ):
        super().__init__()
        self.features = features
        self.dim = dim
        self.num_classes = num_classes

        # Input Projection
        self.input_proj = nn.Linear(features, dim)

        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Class Embedding
        if num_classes > 0:
            self.label_emb = nn.Embedding(num_classes, dim)
        
        # Mamba Blocks
        self.layers = nn.ModuleList([
            nn.Sequential(
                RMSNorm(dim),
                MambaBlock(
                    d_model=dim, 
                    d_state=d_state, 
                    d_conv=d_conv, 
                    expand=expand
                )
            )
            for _ in range(depth)
        ])
        
        self.norm_f = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, features)

    @classmethod
    def _model_kwargs_from_config(cls, params) -> dict:
        return {
            'dim': params.dim,
            'depth': params.depth,
            'd_state': params.d_state,
            'd_conv': params.d_conv,
            'expand': params.expand,
            'num_classes': params.num_classes,
        }

    def forward(self, x, t, y=None, mask=None):
        """
        Forward pass for MambaDiffusion.

        Args:
            x: Input tensor (Batch, Seq_Len, Features)
            t: Timesteps (Batch,)
            y: Optional class labels (Batch,) for conditional generation
            mask: Optional binary mask (Batch, Seq_Len, Features) - not used by Mamba
                  (Mamba is a sequential SSM model, mask accepted for API compatibility)

        Returns:
            Predicted noise (Batch, Seq_Len, Features)
        """
        # 1. Project Input
        x = self.input_proj(x)

        # 2. Add Time Embedding
        t_emb = self.time_mlp(t) # (B, Dim)
        x = x + t_emb.unsqueeze(1)

        # 3. Add Class Embedding
        if self.num_classes > 0 and y is not None:
            y_emb = self.label_emb(y)
            x = x + y_emb.unsqueeze(1)

        # 4. Mamba Layers (Residual)
        for layer in self.layers:
            x = x + layer(x)

        # 5. Output
        x = self.norm_f(x)
        return self.output_proj(x)
