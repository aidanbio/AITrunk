"""
Sampling algorithms by reverse process of diffusion models including DDPM and DDIM.
This is based on the labml_nn.diffusion.stable_diffusion.sampler modules which have implemented the following papers:
DDPM: Ho, Jonathan, et al. "Denoising diffusion probabilistic models." arXiv preprint arXiv:2006.11239 (2020).
DDIM: Yang, Song, et al. "Denoising diffusion implicit models." arXiv preprint arXiv:2106.09623 (2021).
"""

import unittest
import torch
from aitrunk.utils import gather

class DiffusionSampler(object):
    """
    ## Base class for sampling algorithms
    """
    def __init__(self, model=None):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__()

        self.model = model
        self.n_steps = model.n_steps
        self.time_steps = torch.arange(0, self.n_steps, dtype=torch.long)
        self.device = model.device

    def get_eps(self, x, t, c, uncond=None, uncond_scale=1.):
        """
        ## Get $\epsilon(x_t, c)$

        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param t: is $t$ of shape `[batch_size]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param uncond: is the conditional embedding for empty prompt $c_u$
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        """
        # When the scale $s = 1$
        # $$\epsilon_\theta(x_t, c) = \epsilon_\text{cond}(x_t, c)$$
        if uncond is None or uncond_scale == 1.:
            return self.model(x, t, c)

        # Duplicate $x_t$ and $t$
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        # Concatenated $c$ and $c_u$
        c_in = torch.cat([uncond, c])
        # Get $\epsilon_\text{cond}(x_t, c)$ and $\epsilon_\text{cond}(x_t, c_u)$
        e_t_uncond, e_t_cond = self.model(x_in, t_in, c_in).chunk(2)
        # Calculate
        # $$\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$$
        e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)
        return e_t

    @torch.no_grad()
    def sample(self, x_shape, cond, uncond=None, uncond_scale=1.):
        """
        ### Sampling Loop
        :param x_shape: is (channels, height, width) of the samples
        :param cond: is the conditional embeddings $c$, (batch_size, embedding_size)
        :param uncond: is the conditional embedding for empty prompt $c_u$
        :param uncond_scale: is the unconditional guidance scale $s$. eps_theta = e_uc + s*(e_c - e_uc)
        """
        bs = cond.shape[0]
        # Random noisy sample $x_T$
        x = torch.randn((bs, *x_shape), device=self.device)
        # Reverse process to remove noise from $x_T$ to $x_0$
        for i, t in enumerate(reversed(self.time_steps)):
            ts = x.new_full((bs,), t, dtype=torch.long)
            ti = self.n_steps - i - 1
            # Sample $x_{t-1}$
            x = self.p_sample(x, ts, ti, cond=cond, uncond=uncond, uncond_scale=uncond_scale)
            print(f'Reversed to remove noise at {t}')
        return x

    def p_sample(self, xt, ts, ti, cond=None, uncond=None, uncond_scale=1.):
        """
        Sample from ${p_\theta}(x_{t-1}|x_t)$
        :param xt: is the noised sample $x_t$
        :param ts: is the time step $t$, (batch_size,)
        :param ti: is the index of the time step $t$ in scheduled time steps
        :param cond: is the conditional embeddings $c$
        :param uncond: is the conditional embedding for empty prompt $c_u$
        :param uncond_scale: is the unconditional guidance scale $s$. eps_theta = e_uc + s*(e_c - e_uc)
        """
        raise NotImplementedError()


class DDPMSampler(DiffusionSampler):
    def __init__(self, model=None):
        super().__init__(model)

        self.alpha = 1. - model.beta.to(self.device)
        self.alpha_bar = model.alpha_bar.to(self.device)
        self.sigma2 = model.beta.to(self.device)

    @torch.no_grad()
    def p_sample(self, xt, ts, ti, cond=None, uncond=None, uncond_scale=1.):
        # ${\epsilon_\theta}(x_t, t)$
        eps_theta = self.get_eps(x=xt, t=ts, c=cond, uncond=uncond, uncond_scale=uncond_scale)

        alpha_bar = gather(self.alpha_bar, ts)
        alpha = gather(self.alpha, ts)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        if ti == 0:
            return mean

        # $\sigma^2$
        var = gather(self.sigma2, ts)
        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=self.device)
        # Sampling with reparameterization trick
        return mean + (var ** .5) * eps


class DDIMSampler(DiffusionSampler):
    def __init__(self, model=None, n_steps=20, discretize='uniform', eta=0.):
        """
        :param model:is the latent diffusion model
        :param n_steps: is the DDIM sampling steps $S$
        :param discretize: is how to extract $tau$ the time steps. 'uniform' or 'quad'
        :param eta: is used to calculate $sigma_t. $eta=0$ is the original DDPM
        of which the sampling process is deterministic
        """
        super().__init__(model)

        total_steps = model.n_steps
        self.n_steps = n_steps
        # Calculate $tau to be uniformly or quadratically distributed across
        if discretize == 'uniform':
            c = total_steps // self.n_steps
            self.time_steps = torch.arange(0, total_steps, c) + 1
        elif discretize == 'quad':
            end = torch.sqrt(total_steps * .8).to(torch.float32)
            self.time_steps = (torch.linspace(0, end, self.n_steps) ** 2).to(torch.long) + 1
        else:
            raise ValueError(f'Unknown discretize {discretize}')

        alpha_bar = self.model.alpha_bar
        self.alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
        self.alpha_sqrt = torch.sqrt(self.alpha)
        self.alpha_prev = torch.cat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]])
        self.sigma = (eta * ((1 - self.alpha_prev) / (1 - self.alpha) *
                             (1 - self.alpha / self.alpha_prev)) ** .5)
        self.sqrt_one_minus_alpha = (1. - self.alpha) ** .5

    @torch.no_grad()
    def p_sample(self, xt, ts, ti, cond=None, uncond=None, uncond_scale=1.):
        et = self.get_eps(x=xt, t=ts, c=cond, uncond=uncond, uncond_scale=uncond_scale)
        x_prev, _ = self._get_x_prev_and_pred_x0(xt, ti, et)
        return x_prev

    def _get_x_prev_and_pred_x0(self, xt, ti, et):
        """
        Calculate $x_{t-1}$ and predict $x_0$ from $x_t$ and $e_theta$
        :param xt: is the noised sample $x_t$
        :param ti: is the index of the time step $t$ in scheduled time steps
        :param et: is the e_theta at time $t$
        """
        alpha = self.alpha[ti]
        alpha_prev = self.alpha_prev[ti]
        sigma = self.sigma[ti]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha[ti]
        pred_x0 = (xt - sqrt_one_minus_alpha * et) / (alpha ** 0.5)
        # Direction pointing to xt
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * et
        if sigma == 0.:
            noise = 0.
        else:
            noise = torch.randn(xt.shape, device=xt.device)
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        return x_prev, pred_x0


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
