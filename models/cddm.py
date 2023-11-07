import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class cDDM(nn.Module):
    def __init__(self, denoising_net, denoising_target="eps", beta1=1e-4, betaT=0.02, T=1000, criterion=nn.MSELoss()):
        '''
        General conditional denoising diffusion model. Sampling method needs to be defined in subclasses.
        '''
        super().__init__()
        self.denoising_net = denoising_net # denoising U-net, forward method outputs an estimate of <denoising_target>
        if denoising_target in ["eps", "y"]:
            self.denoising_target = denoising_target
        else:
            raise ValueError(f"{denoising_target} is not a valid denoising target. Use either 'eps' or 'y'.")

        # register_buffer allows us to freely access these tensors by name
        for k, v in noise_schedules(beta1, betaT, T).items():
            self.register_buffer(k, v, persistent=False)

        self.T = T
        self.criterion = criterion

    def forward_prediction(self, y, x, seed=None):
        '''
        Used by the forward method, and useful for visualizing the process.

        Sample random time step t.
        Sample y_t through forward process (using reparametrization with variable eps).
        Estimates <denoising_target> (either eps or y) from (y_t, x, t) using denoising_net.
        Returns t, eps, y_t, and estimated_target
        '''
        device = y.device
        generator = torch.Generator(device)
        if seed is None:
            seed = generator.seed() # get a random seed : default seed of generator is deterministic
        generator.manual_seed(seed)
        # Sample timestep uniformly
        t = torch.randint(1, self.T + 1, (y.shape[0],), device=device, generator=generator)
        # Sample y_t from forward process through reparametrization
        eps = torch.randn(y.shape, device=device, generator=generator)  # eps ~ N(0, 1)
        y_t = y*self.sqrt_gamma[t,None,None,None] + eps*self.sqrt_one_minus_gamma[t,None,None,None]
        # Estimate denoising target using denoising_net
        estimated_target = self.denoising_net(torch.cat((y_t, x), dim=1), t / self.T)
        return t, eps, y_t, estimated_target

    def forward(self, y, x, seed=None):
        '''
        Run forward_prediction method to get t, eps, y_t, and an estimate of the target.
        Compute estimated eps from estimated target.
        Returns the criterion loss between real and estimated eps.
        '''
        t, eps, y_t, estimated_target = self.forward_prediction(y, x, seed)
        estimated_eps = self.get_eps_from_target(estimated_target, y_t, t)
        return self.criterion(eps, estimated_eps)
    
    def get_eps_from_target(self, target, y_t, t):
        if self.denoising_target=="eps":
            eps = target 
        elif self.denoising_target=="y":
            eps = (y_t - target*self.sqrt_gamma[t,None,None,None])/self.sqrt_one_minus_gamma[t,None,None,None]
        return eps
    
    def get_y_from_target(self, target, y_t, t):
        if self.denoising_target=="eps":
            y = (y_t - target*self.sqrt_one_minus_gamma[t,None,None,None])/self.sqrt_gamma[t,None,None,None]
        elif self.denoising_target=="y":
            y = target
        return y


class cDDPM(cDDM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, x, return_trajectory=False, seed=None):
        '''x is a BxCxHxW tensor. Returns a tensor of the same shape.'''
        # Get device and B (number of samples) from x
        device = x.device
        n_sample = x.shape[0]
        # Set generator and (optional) seed for reproducibility
        generator = torch.Generator(device)
        if seed is None:
            seed = generator.seed() # get a random seed : default seed of generator is deterministic
        generator.manual_seed(seed)
        # Storing intermediate results in case return_trajectory=True
        y_noisy = [] # list containing all y_t's
        y_estimates = [] # list containing all intermediate estimates of y_0

        # DDPM sampling
        y_t = torch.randn(x.shape, generator=generator, device=device) # Initialize y_T with pure noise
        for t in tqdm(torch.arange(self.T, 0, -1), desc="Sampling..."):
            # Compute estimate of eps
            tt = t.repeat(n_sample)
            estimated_target = self.denoising_net(torch.cat((y_t, x), 1), (tt / self.T).to(device))
            # Compute estimate of y_0 (useful only if return_trajectory is True)
            estimated_y = self.get_y_from_target(estimated_target, y_t, tt)
            estimated_eps = self.get_eps_from_target(estimated_target, y_t, tt)
            # Store intermediate results
            y_noisy.append(y_t)
            y_estimates.append(estimated_y)
            # Compute next step of reverse process
            z = torch.randn(x.shape, generator=generator, device=device) if t > 1 else 0
            y_t = self.oneover_sqrta[t] * (y_t - estimated_eps * self.beta_over_sqrt_one_minus_gamma[t]) + self.sqrtbeta[t] * z # sigma is sqrt_beta here
        y_noisy.append(y_t) # y_0 
        y_estimates.append(y_estimates) # not really useful as it is the same element as the prev one, but to make size consistent with y_noisy

        if return_trajectory:
            return torch.cat(y_noisy), torch.cat(y_estimates)
        else:
            return y_t

# OUTDATED CODE. DDIM sampling and DDPM sampling for SR3 (gamma instead of t, and inverts y and x for the unet)

# class cDDIM(cDDM):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def sample(self, x_cond, tau: torch.Tensor, return_trajectory=False, seed=None):
#         '''x_cond is a BxCxHxW tensor. Returns a tensor of the same shape.'''
#         # Get device and B (number of samples) from x_cond
#         device = x_cond.device
#         n_sample = x_cond.shape[0]
#         # Set generator and (optional) seed for reproducibility
#         generator = torch.Generator(device)
#         if seed is not None:
#             generator.manual_seed(seed)
#         # Storing intermediate results in case return_trajectory=True
#         x = [] # vector containing all x_i's
#         x_0_estimates = [] # vector containing all intermediate estimates of x0

#         # DDIM sampling
#         x_ti = torch.randn(x_cond.shape, generator=generator, device=device) # Initialize x_T with pure noise
#         for i in tqdm(reversed(range(len(tau))), desc="Sampling..."):
#             ti = tau[i]
#             # Compute estimate of eps
#             eps_ti = self.eps_model(torch.cat((x_ti, x_cond), 1), (ti / self.n_T).to(device).repeat(n_sample, 1))
#             # Compute estimate of x_0
#             x_0_ti = (x_ti - eps_ti*self.sqrtmab[ti])/self.sqrtab[ti]
#             # Store intermediate results
#             x.append(x_ti)
#             x_0_estimates.append(x_0_ti)
#             # Compute next step of reverse process
#             if i > 0:
#                 tim = tau[i-1]
#                 x_ti = self.sqrtab[tim]*x_0_ti + self.sqrtmab[tim]*eps_ti
#             else:
#                 x_ti = x_0_ti
#         x.append(x_ti)
#         x_0_estimates.append(x_0_ti) # not really useful as x_0_0 = x_0_1, but to make size consistent with x

#         if return_trajectory:
#             return torch.cat(x), torch.cat(x_0_estimates)
#         else:
#             return x_ti
        
# class SR3(cDDM):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def sample(self, x_cond, return_trajectory=False, seed=None):
#         '''x_cond is a BxCxHxW tensor. Returns a tensor of the same shape.'''
#         # Get device and B (number of samples) from x_cond
#         device = x_cond.device
#         n_sample = x_cond.shape[0]
#         # Set generator and (optional) seed for reproducibility
#         generator = torch.Generator(device)
#         if seed is None:
#             seed = generator.seed() # get a random seed : default seed of generator is deterministic
#         generator.manual_seed(seed)
#         # Storing intermediate results in case return_trajectory=True
#         x = [] # vector containing all x_i's
#         x_0_estimates = [] # vector containing all intermediate estimates of x0

#         # DDPM sampling
#         x_t = torch.randn(x_cond.shape, generator=generator, device=device) # Initialize x_T with pure noise
#         for t in tqdm(torch.arange(self.n_T, 0, -1), desc="Sampling..."):
#             # Compute estimate of eps
#             noise_level = self.sqrtab[t].to(device).repeat(n_sample, 1)
#             eps_t = self.eps_model(torch.cat((x_cond, x_t), 1), noise_level)
#             # Compute estimate of x_0 (useful only if return_trajectory is True)
#             x_0_t = (x_t - eps_t * self.sqrtmab[t])/self.sqrtab[t]
#             # Store intermediate results
#             x.append(x_t)
#             x_0_estimates.append(x_0_t)
#             # Compute next step of reverse process
#             z = torch.randn(x_cond.shape, generator=generator, device=device) if t > 1 else 0
#             x_t = self.oneover_sqrta[t] * (x_t - eps_t * self.beta_over_sqrtmab[t]) + self.sqrtbeta[t] * z # sigma is sqrt_beta here
#         x.append(x_t)
#         x_0_estimates.append(x_0_t)

#         if return_trajectory:
#             return torch.cat(x), torch.cat(x_0_estimates)
#         else:
#             return x_t


def noise_schedules(beta1, betaT, T):
    '''
    Returns pre-computed schedules. Uses DDPM paper notation
    '''
    assert beta1 < betaT < 1.0, "beta1 and betaT must be in (0, 1)"
    # linear noise schedule
    beta = torch.linspace(beta1, betaT, T)
    beta = torch.cat([torch.zeros(1), beta]) # beta[0] is never accessed; but needed to make consistent indexing (beta[1] = beta1])
    sqrtbeta = torch.sqrt(beta)
    alpha = 1 - beta
    log_alpha = torch.log(alpha)
    gamma = torch.cumsum(log_alpha, dim=0).exp()

    sqrt_gamma = torch.sqrt(gamma)
    oneover_sqrta = 1 / torch.sqrt(alpha)

    sqrt_one_minus_gamma = torch.sqrt(1 - gamma)
    beta_over_sqrt_one_minus_gamma = beta / sqrt_one_minus_gamma

    return {
        "alpha": alpha,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrtbeta": sqrtbeta,  # \sqrt{\beta_t}
        "gamma": gamma,  # \bar{\alpha_t}
        "sqrt_gamma": sqrt_gamma,  # \sqrt{\bar{\alpha_t}}
        "sqrt_one_minus_gamma": sqrt_one_minus_gamma,  # \sqrt{1-\bar{\alpha_t}}
        "beta_over_sqrt_one_minus_gamma": beta_over_sqrt_one_minus_gamma,  # (\beta_t)/\sqrt{1-\bar{\alpha_t}}
    }

def get_tau(dim, T=1000):
    ii = torch.arange(1, dim+1)
    c = T/dim
    tau = torch.floor(c*ii).type(torch.int16)
    return tau