import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class cDM(nn.Module):
    def __init__(self, denoising_net, denoising_target, loss_weighting, beta1=1e-4, betaT=0.02, T=1000):
        '''
        General conditional denoising diffusion model.
        '''
        super().__init__()
        self.denoising_net = denoising_net # denoising U-net, forward method outputs an estimate of <denoising_target>
        if denoising_target in ["eps", "y", "both"]:
            self.denoising_target = denoising_target
        else:
            raise ValueError(f"{denoising_target} is not a valid denoising target. Use either 'eps' or 'y'.")
        if loss_weighting in ["uniform", "SNR", "minSNR-5", "maxSNR-1"]:
            self.loss_weighting = loss_weighting
        else:
            raise ValueError(f"{loss_weighting} is not a valid loss weighting strategy.")

        # register_buffer allows us to freely access these tensors by name
        for k, v in noise_schedules(beta1, betaT, T).items():
            self.register_buffer(k, v, persistent=False)

        self.T = T

    def forward_prediction(self, y, x, seed=None, t=None):
        '''
        Used by the forward method, and useful for visualizing the process.

        Sample random time step t (if t is None)
        Sample y_t through diffusion process (using reparametrization with variable eps).
        Estimates <denoising_target> (either eps or y) from (y_t, x, t) using denoising_net.
        Returns t, eps, y_t, and estimated_target
        '''
        B = y.shape[0]
        device = y.device
        generator = torch.Generator(device)
        if seed is None:
            # Get a random seed : default seed of generator is deterministic
            seed = generator.seed() 
        generator.manual_seed(seed)
        if t is None:
            # Sample timestep uniformly
            t = torch.randint(1, self.T + 1, (B,), device=device, generator=generator)
        else:
            # t is a tensor of size B given as input
            t = t.to(device)
        # Sample y_t from diffusion process through reparametrization
        eps = torch.randn(y.shape, device=device, generator=generator)  # eps ~ N(0, 1)
        y_t = y*self.sqrt_gamma[t,None,None,None] + eps*self.sqrt_one_minus_gamma[t,None,None,None]
        # Estimate denoising target using denoising_net
        estimated_target = self.denoising_net(torch.cat((y_t, x), dim=1), t / self.T)
        return t, eps, y_t, estimated_target

    def forward(self, y, x, seed=None):
        '''
        Run forward_prediction method to get t, eps, y_t, and an estimate of the target.
        Compute estimated y from estimated target.
        Returns the loss between real and estimated y.
        '''
        t, _, y_t, estimated_target = self.forward_prediction(y, x, seed)
        estimated_y = self.get_y_from_target(estimated_target, y_t, t)
        loss = self.compute_loss(y, estimated_y, t)
        return loss
    
    def compute_loss(self, y, estimated_y, t):
        loss_per_pixel = nn.MSELoss(reduction='none')(y, estimated_y) # BxCxHxW tensor
        loss_per_image = torch.mean(loss_per_pixel, dim=(1,2,3)) # B tensor
        if self.loss_weighting=="uniform":
            weight = 1
        if self.loss_weighting=="SNR":
            weight = self.gamma[t]/(1-self.gamma[t])
        if self.loss_weighting=="minSNR-5":
            weight = torch.clip(self.gamma[t]/(1-self.gamma[t]), max=5)
        if self.loss_weighting=="maxSNR-1":
            weight = torch.clip(self.gamma[t]/(1-self.gamma[t]), min=1)
        loss = torch.mean(loss_per_image*weight)
        return loss
    
    def get_eps_from_target(self, target, y_t, t):
        if self.denoising_target=="eps":
            eps = target 
        elif self.denoising_target=="y":
            eps = (y_t - target*self.sqrt_gamma[t,None,None,None])/self.sqrt_one_minus_gamma[t,None,None,None]
        elif self.denoising_target=="both":
            eps_explicit = target[:,0:1]
            y = target[:,1:2]
            eps_implicit = (y_t - y*self.sqrt_gamma[t,None,None,None])/self.sqrt_one_minus_gamma[t,None,None,None]
            eps = (1-self.gamma[t,None,None,None])*eps_implicit + self.gamma[t,None,None,None]*eps_explicit
        return eps
    
    def get_y_from_target(self, target, y_t, t):
        if self.denoising_target=="eps":
            y = (y_t - target*self.sqrt_one_minus_gamma[t,None,None,None])/self.sqrt_gamma[t,None,None,None]
        elif self.denoising_target=="y":
            y = target
        elif self.denoising_target=="both":
            y_explicit = target[:,1:2]
            eps = target[:,0:1]
            y_implicit = (y_t - eps*self.sqrt_one_minus_gamma[t,None,None,None])/self.sqrt_gamma[t,None,None,None]
            y = (1-self.gamma[t,None,None,None])*y_explicit + self.gamma[t,None,None,None]*y_implicit
        return y
    
    def sample(self, x, n_steps=None, sampling_mode='DDPM', return_trajectory=False, seed=None):
        '''x is a BxCxHxW tensor. By default, returns a tensor of the same shape.
        If return_trajectory=True, returns two tensors of size ((n_steps+1)*B)xCxHxW.'''
        # Get device and B (number of samples) from x
        device = x.device
        n_sample = x.shape[0]
        # Set generator and (optional) seed for reproducibility
        generator = torch.Generator(device)
        if seed is None:
            seed = generator.seed() # get a random seed : default seed of generator is deterministic
        generator.manual_seed(seed)
        # Make timesteps vector (subset of [1,...,self.T])
        if n_steps is None:
            n_steps = self.T
        space = torch.linspace(0, 1, n_steps+1)[1:] # remove first element (which is 0)
        timesteps = torch.floor(self.T*space).type(torch.int64)
        # Storing intermediate results in case return_trajectory=True
        list_y_t = [] # list containing all y_t's
        list_estimated_y = [] # list containing all intermediate estimates of y_0
        # DDPM sampling
        y_t = torch.randn(x.shape, generator=generator, device=device) # Initialize y_T with pure noise
        for i in tqdm(reversed(range(n_steps)), total=n_steps, desc="Sampling..."):
            t = timesteps[i]
            # Compute estimates of y and eps
            tt = t.repeat(n_sample)
            estimated_target = self.denoising_net(torch.cat((y_t, x), 1), (tt / self.T).to(device))
            estimated_y = self.get_y_from_target(estimated_target, y_t, tt)
            estimated_eps = self.get_eps_from_target(estimated_target, y_t, tt)
            # Store intermediate results (only useful if return_trajectory=True)
            list_y_t.append(y_t) 
            list_estimated_y.append(estimated_y)
            # Compute next step of reverse process
            y_t = self.reverse_step(estimated_y, estimated_eps, timesteps, i, sampling_mode, generator)
        list_y_t.append(y_t) # y_0 
        list_estimated_y.append(estimated_y) # not really useful as it is the same element as the prev one, but to make size consistent with y_noisy

        if return_trajectory:
            return torch.cat(list_y_t), torch.cat(list_estimated_y)
        else:
            return y_t
        
    # Below are the modifications needed for sampling pretrained SR3 (gamma instead of t, and inverts y and x for the unet)
    # noise_level = self.sqrt_gamma[t].repeat(n_sample, 1)
    # estimated_target = self.denoising_model(torch.cat((x, y_t), 1), noise_level.to(device))

    def reverse_step(self, estimated_y, estimated_eps, timesteps, i, sampling_mode, generator):
        if i != 0:
            t  = timesteps[i]
            tm = timesteps[i-1]
            if sampling_mode=='DDPM':
                z = torch.randn(estimated_y.shape, generator=generator, device=estimated_y.device)
                sigma = torch.sqrt((1-self.gamma[tm])/(1-self.gamma[t]))*self.sqrtbeta[t]
                y_t = self.sqrt_gamma[tm]*estimated_y + torch.sqrt(1-self.gamma[tm]-sigma**2)*estimated_eps + sigma*z
            if sampling_mode=='DDIM':
                y_t = self.sqrt_gamma[tm]*estimated_y + self.sqrt_one_minus_gamma[tm]*estimated_eps
        else:
            y_t = estimated_y
        return y_t 

class TADM(cDM):
    def __init__(self, denoising_net, task_net, task_weight, denoising_target, loss_weighting, **kwargs):
        super().__init__(denoising_net, denoising_target, loss_weighting, **kwargs)
        self.task_net = task_net
        self.task_weight = task_weight

    def forward(self, y, x, seed=None):
        '''
        Runs forward_prediction method to get t, eps, y_t, and an estimate of the target.
        Computes and returns both denoising and task loss.
        '''
        t, eps, y_t, estimated_target = self.forward_prediction(y, x, seed)
        estimated_y = self.get_y_from_target(estimated_target, y_t, t)
        denoising_loss = self.compute_loss(y, estimated_y, t)

        if self.task_weight == 0: # to save time, avoids unnecessary computations
            task_loss = torch.zeros_like(denoising_loss)
        else:
            # Task network was trained on [0, 1] images with no match means. Hardcoding this here until better solution
            coeffs_sted = (0.0850, 0.1107)
            estimated_y = estimated_y*coeffs_sted[1] + coeffs_sted[0]
            y = y*coeffs_sted[1] + coeffs_sted[0]
            # Need to scale inputs for the segmentation network
            factor = 5.684 # 0.8*2.25 (0.8*max(dataset)) divided by 0.3166458 (match means avg correction)
            self.task_net.eval()
            with torch.no_grad():
                y_task = self.task_net(y/factor)
            estimated_y_task = self.task_net(estimated_y/factor)

            task_loss = self.task_weight*self.compute_loss(y_task, estimated_y_task, t) # also uses weighted MSE

        return denoising_loss, task_loss


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

def get_tau(n_steps, T=1000):
    ii = torch.arange(1, n_steps+1)
    c = T/n_steps
    tau = torch.floor(c*ii).type(torch.int16)
    return tau