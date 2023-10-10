import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class cDDM(nn.Module):
    def __init__(self, eps_model: nn.Module, betas: tuple[float, float], n_T: int, criterion: nn.Module = nn.MSELoss()):
        '''
        General conditional denoising diffusion model. Sampling method needs to be defined in subclasses.
        '''
        super().__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name
        for k, v in noise_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v, persistent=False)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x, x_cond):
        '''
        Sample random time step t.
        Sample x_t through forward process (using reparametrization with variable eps).
        Estimates eps from (x_t, x_cond, t) using eps_model.
        Returns the MSE between real and estimated eps.
        '''
        # Sample timestep uniformly
        t = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        # Sample x_t from forward process through reparametrization
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        x_t = self.sqrtab[t] * x + self.sqrtmab[t] * eps
        # Estimate eps using eps_model
        estimated_eps = self.eps_model(torch.cat((x_t, x_cond), 1), t / self.n_T)

        return self.criterion(eps, estimated_eps)


class cDDPM(cDDM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, x_cond, return_trajectory=False, seed=None):
        '''x_cond is a BxCxHxW tensor. Returns a tensor of the same shape.'''
        # Get device and B (number of samples) from x_cond
        device = x_cond.device
        n_sample = x_cond.shape[0]
        # Set generator and (optional) seed for reproducibility
        generator = torch.Generator(device)
        if seed is not None:
            generator.manual_seed(seed)
        # Storing intermediate results in case return_trajectory=True
        x = [] # vector containing all x_i's
        x_0_estimates = [] # vector containing all intermediate estimates of x0

        # DDPM sampling
        x_i = torch.randn(x_cond.shape, generator=generator, device=device) # Initialize x_T with pure noise
        for i in torch.arange(self.n_T, 0, -1):
            # Compute estimate of eps
            eps_i = self.eps_model(torch.cat((x_i, x_cond), 1), (i / self.n_T).to(device).repeat(n_sample, 1))
            # Compute estimate of x_0 (useful only if return_trajectory is True)
            x_0_i = (x_i - eps_i * self.sqrtmab[i])/self.sqrtab[i]
            # Compute next step of reverse process
            z = torch.randn(x_cond.shape, generator=generator, device=device) if i > 1 else 0
            x_i = (self.oneover_sqrta[i] * (x_i - eps_i * self.beta_over_sqrtmab[i]) + self.sqrtbeta[i] * z) # sigma is sqrt_beta here
            # Store intermediate results
            x.append(x_i)
            x_0_estimates.append(x_0_i)

        if return_trajectory:
            return torch.cat(x), torch.cat(x_0_estimates)
        else:
            return x_i


class cDDIM(cDDM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, x_cond, tau: torch.Tensor, return_trajectory=False, seed=None):
        '''x_cond is a BxCxHxW tensor. Returns a tensor of the same shape.'''
        # Get device and B (number of samples) from x_cond
        device = x_cond.device
        n_sample = x_cond.shape[0]
        # Set generator and (optional) seed for reproducibility
        generator = torch.Generator(device)
        if seed is not None:
            generator.manual_seed(seed)
        # Storing intermediate results in case return_trajectory=True
        x = [] # vector containing all x_i's
        x_0_estimates = [] # vector containing all intermediate estimates of x0

        # DDIM sampling
        x_ti = torch.randn(x_cond.shape, generator=generator, device=device) # Initialize x_T with pure noise
        for i in reversed(range(len(tau))):
            ti = tau[i]
            # Compute estimate of eps
            eps_ti = self.eps_model(torch.cat((x_ti, x_cond), 1), (ti / self.n_T).to(device).repeat(n_sample, 1))
            # Compute estimate of x_0
            x_0_ti = (x_ti - eps_ti*self.sqrtmab[ti])/self.sqrtab[ti]
            # Compute next step of reverse process
            if i > 0:
                tim = tau[i-1]
                x_ti = self.sqrtab[tim]*x_0_ti + self.sqrtmab[tim]*eps_ti
            else:
                x_ti = x_0_ti
            # Store intermediate results
            x.append(x_ti)
            x_0_estimates.append(x_0_ti)

        if return_trajectory:
            return torch.cat(x), torch.cat(x_0_estimates)
        else:
            return x_ti


def noise_schedules(beta1, betaT, T):
    '''
    Returns pre-computed schedules. Uses DDPM paper notation
    '''
    assert beta1 < betaT < 1.0, "beta1 and betaT must be in (0, 1)"

    beta = torch.linspace(beta1, betaT, T)
    beta = torch.cat([torch.zeros(1), beta]) # beta[0] is never accessed; but needed to make consistent indexing (beta[1] = beta1])
    sqrtbeta = torch.sqrt(beta)
    alpha = 1 - beta
    log_alpha = torch.log(alpha)
    alphabar = torch.cumsum(log_alpha, dim=0).exp()

    sqrtab = torch.sqrt(alphabar)
    oneover_sqrta = 1 / torch.sqrt(alpha)

    sqrtmab = torch.sqrt(1 - alphabar)
    beta_over_sqrtmab = beta / sqrtmab

    return {
        "alpha": alpha,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrtbeta": sqrtbeta,  # \sqrt{\beta_t}
        "alphabar": alphabar,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "beta_over_sqrtmab": beta_over_sqrtmab,  # (\beta_t)/\sqrt{1-\bar{\alpha_t}}
    }

def get_tau(dim, n_T=1000):
    ii = torch.arange(1, dim+1)
    c = n_T/dim
    tau = torch.floor(c*ii).type(torch.int16)
    return tau