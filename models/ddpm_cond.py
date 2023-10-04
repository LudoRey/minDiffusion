from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .ddpm import DDPM


class cDDPM(DDPM):
    def __init__(
            self,
            eps_model: nn.Module,
            betas: Tuple[float, float],
            n_T: int,
            criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__(eps_model, betas, n_T, criterion)

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(torch.cat((x_t, x_cond), 1), _ts / self.n_T))

    def sample(self, x_cond: torch.Tensor, device, return_trajectory=False) -> torch.Tensor:
        n_sample = x_cond.shape[0]
        size = x_cond.shape[1:]
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # storing intermediate results in case return_trajectory=True
        x = [] # vector containing all x_i's
        x0_estimates = [] # vector containing all intermediate estimates of x0

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(
                torch.cat((x_i, x_cond), 1), torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
            )
            x0_i = (x_i - eps * self.sqrtmab[i])/self.alphabar_t[i]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            
            x.append(x_i)
            x0_estimates.append(x0_i)

        if return_trajectory:
            return torch.cat(x), torch.cat(x0_estimates)
        else:
            return x_i