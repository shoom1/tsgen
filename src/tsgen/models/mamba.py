import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tsgen.models.base_model import GenerativeModel
from tsgen.models.embeddings import SinusoidalPositionEmbeddings

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class MambaBlock(nn.Module):
    """
    A pure PyTorch implementation of the Mamba block (Selective SSM).
    
    This implementation avoids the specialized CUDA kernels of the official 'mamba-ssm'
    package to ensure compatibility with CPU and standard GPU environments (like Mac).
    It uses a sequential recurrence which is slower than the parallel scan but
    functionally equivalent for training/inference.
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

    def ssm(self, x):
        """
        Runs the SSM recurrence:
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t
        """
        (batch, seq_len, d_inner) = x.shape
        
        # 1. Project x to Delta, B, C
        # x_dbl: (B, L, dt_rank + 2*d_state)
        x_dbl = self.x_proj(x) 
        
        # Split into delta, B, C
        delta, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # delta: (B, L, dt_rank) -> (B, L, d_inner)
        delta = F.softplus(self.dt_proj(delta))
        
        # B, C: (B, L, d_state)
        
        # 2. Discretize A -> A_bar
        # A: (d_inner, d_state)
        A = -torch.exp(self.A_log.float())  
        
        # We need to broadcast terms for the recurrence
        # We'll do a sequential loop for simplicity and compatibility (Pure PyTorch)
        
        ys = []
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        
        for t in range(seq_len):
            # Delta_t: (B, d_inner)
            dt = delta[:, t, :]
            
            # A_bar = exp(Delta * A)
            # (B, d_inner, 1) * (1, d_inner, d_state) -> (B, d_inner, d_state)
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            
            # B_bar = Delta * B
            # (B, d_inner) * (B, d_state) -> (B, d_inner, d_state) (outer product effectively per batch item)
            # Actually standard Mamba formulation: B is (B, L, N)
            # We want (B, D, N) where D is inner dim, N is state dim
            # But B is shared across D in the simplest simplified view or projected? 
            # In official Mamba: B is (B, L, N), we broadcast to D.
            
            # Current slice:
            Bt = B[:, t, :] # (B, N)
            xt = x[:, t, :] # (B, D)
            
            # dB = dt * B
            # (B, D, 1) * (B, 1, N) -> (B, D, N)
            dB = torch.einsum("bd,bn->bdn", dt, Bt)
            
            # Recurrence: h = dA * h + dB * x
            # x is (B, D), we need (B, D, N) so we scale dB by x
            # (B, D, N) * (B, D, 1)
            h = dA * h + dB * xt.unsqueeze(-1)
            
            # y = C * h
            # C is (B, L, N) -> Ct (B, N)
            Ct = C[:, t, :]
            # (B, D, N) * (B, 1, N) -> sum over N -> (B, D)
            y = torch.einsum("bdn,bn->bd", h, Ct)
            
            ys.append(y)
            
        y = torch.stack(ys, dim=1) # (B, L, D)
        
        return y + x * self.D

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

class MambaDiffusion(GenerativeModel):
    """
    Diffusion Model using Mamba (S4) backbone.
    """
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

    def forward(self, x, t, y=None):
        # x: (B, L, Features)
        
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
