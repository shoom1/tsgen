import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

class DiffusionUtils:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def get_time_steps(self, num_inference_steps):
        # Create a list of evenly spaced numbers from 0 to T-1 then reverse it
        c_steps = torch.linspace(0, self.T - 1, num_inference_steps + 1).long().to(self.device)
        return torch.flip(c_steps, [0])

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, y=None):
        betas_t = self.betas[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None]
        
        predicted_noise = model(x, t, y)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=1, y=None):
        model.eval()
        x = torch.randn((batch_size, image_size[0], image_size[1])).to(self.device)
        for i in tqdm(reversed(range(0, self.T)), desc="DDPM Sampling"):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, i, y)
        return x

    @torch.no_grad()
    def ddim_sample(self, model, image_size, batch_size=1, num_inference_steps=50, y=None):
        model.eval()
        # Initial noise sample
        x = torch.randn((batch_size, image_size[0], image_size[1])).to(self.device)
        
        # Get DDIM time steps
        time_steps = self.get_time_steps(num_inference_steps)
        
        # Iterate over the time steps
        for i, t_step in enumerate(tqdm(time_steps, desc="DDIM Sampling")):
            if t_step == 0:
                # This is the last step, handle it as x0
                continue
            
            # Current time step (t) and previous time step (t_prev)
            t = torch.full((batch_size,), t_step.item(), device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t, y)
            
            # Calculate x_0_pred
            alpha_prod_t = self.alphas_cumprod[t_step.item()]
            sqrt_one_minus_alpha_prod_t = self.sqrt_one_minus_alphas_cumprod[t_step.item()]
            
            x_0_pred = (x - sqrt_one_minus_alpha_prod_t * predicted_noise) / torch.sqrt(alpha_prod_t)
            
            # Get the previous time step in the DDIM schedule
            t_prev_idx = i + 1
            if t_prev_idx < len(time_steps):
                t_prev = time_steps[t_prev_idx].item()
            else:
                t_prev = 0 # Final step
            
            alpha_prod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)
            
            # Coefficients for x_{t-1} calculation
            coeff_epsilon = torch.sqrt(1.0 - alpha_prod_t_prev)
            coeff_x0 = torch.sqrt(alpha_prod_t_prev)
            
            # No added noise (sigma_t = 0 for deterministic DDIM)
            x = coeff_x0 * x_0_pred + coeff_epsilon * predicted_noise
            
        return x

