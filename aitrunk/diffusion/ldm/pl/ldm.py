"""
Pytorch Lightning implementation of Latent Diffusion Model(LDM) from the paper:
Paper: Robin Rombach, et. al., High-Resolution Image Synthesis with Latent Diffusion Models, arXiv, 2022, http://arxiv.org/abs/2112.10752
This implementation is based on labml.ai's implementations:
https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion
"""
from typing import Any

import unittest
from collections import OrderedDict

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from aitrunk.diffusion.ldm.unet import UNetModel
from aitrunk.utils import gather
from deepspeed.ops.adam import DeepSpeedCPUAdam
import deepspeed

class LDM(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()

        self.eps_model = UNetModel(**config['eps_model'])
        self.uncond_scale = config.get('uncond_scale', 7.5)
        self.n_steps = config.get('n_steps', 1000)
        beta_start = config.get('beta_start', 0.00085)
        beta_end = config.get('beta_end', 0.0120)

        # Create β1, β2, …, βT linearly increasing variance schedule
        self.beta = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, self.n_steps).to(self.device) ** 2
        alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(alpha, dim=0)
        # σ^2 = β
        self.sigma2 = self.beta

    def training_step(self, batch, batch_idx):
        return self._loss_and_log(batch, log_prefix='train')

    def validation_step(self, batch, batch_idx):
        return self._loss_and_log(batch, log_prefix='val')

    def configure_callbacks(self):
        pass

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=2e-4)
        return optimizer

    # def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
    #     print(f'>>>Batch {batch_idx} end. Call get_accelerator().empty_cache()')
    #     get_accelerator().empty_cache()

    def create_sampler(self, name='ddim'):
        pass

    def _loss_and_log(self, batch, log_prefix='train'):
        loss = self._get_loss(x0=batch['input'], c=batch['cond'], uc=batch['uncond'])
        self.log(f'{log_prefix}.loss', loss, prog_bar=True)
        return loss

    def _get_loss(self, x0, noise=None, c=None, uc=None):
        """
        Compute loss between source noise from N(0, I) and predicted noise from x0 to xt
        with scaled conditional embedding c and cu
        :param x0: source data
        :param noise: source noise
        :param c: is conditional embedding of shape (batch_size, embedding_size)
        :param uc: is the unconditional embedding for empty prompt, of shape (batch_size, embedding_size)
        """
        batch_size = x0.shape[0]

        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x0)
        # Sampling $x_t \sim \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon_0$
        xt = self._q_sample(x0, t, noise=noise)
        # Get ${\epsilon_\theta}(x_t, t, c, uc)$, where c = $\tau(y)$ and uc = $\tau("")$
        eps_theta = self._get_eps(xt, t, c, uc)

        # MSE loss
        return F.mse_loss(noise, eps_theta)

    def _get_eps(self, x, t, c, uc):
        """
        :param x: is xt of shape (batch_size, n_channels, heights, widths)
        :param t: is t of shape (batch_size,)
        :param c: is conditional embedding of shape (batch_size, embedding_size)
        :param uc: is the conditional embedding for empty prompt, of shape (batch_size, embedding_size)
        """
        if uc is None or self.uncond_scale == 1.:
            return self.eps_model(x, t, c)

        # Duplicate x and t for cu
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([uc, c])

        # Get noise from eps_model
        eps_uc, eps_c = self.eps_model(x_in, t_in, c_in).chunk(2)
        # Unconditional guidance scale for eps_theta
        return eps_uc + self.uncond_scale * (eps_c - eps_uc)

    def _q_sample(self, x0, t, noise=None):
        """
        Sample xt from $q(x_t|x_0)$ with reparameterization trick
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # Get encoder transition $q(x_t|x_0)$ distribution
        self.alpha_bar = self.alpha_bar.to(self.device)
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)

        # Reparameterization trick
        return mean + (var ** 0.5) * noise

    # def generate_sample(self, n=10, sample_size=(1, 32, 32)):
    #
    #     # Random noisy sample $x_T$
    #     x = torch.randn((n, *sample_size), device=self.device)
    #     # Reverse process to remove noise from $x_T$ to $x_0$
    #     for t in reversed(range(self.n_steps)):
    #         x = self.p_sample(x, x.new_full((n,), t, dtype=torch.long))
    #         print(f'Reversed to remove noise at {t}')
    #     return x
    #
    # def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
    #     """
    #     Sample from ${p_\theta}(x_{t-1}|x_t)$
    #     :param xt: noised sample $x_t$
    #     :param t: time step $t$
    #     """
    #     self.alpha = self.alpha.to(self.device)
    #     self.alpha_bar = self.alpha_bar.to(self.device)
    #     self.sigma2 = self.sigma2.to(self.device)
    #
    #     # ${\epsilon_\theta}(x_t, t)$
    #     eps_theta = self.eps_model(xt, t)
    #     alpha_bar = gather(self.alpha_bar, t)
    #     alpha = gather(self.alpha, t)
    #     # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
    #     eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
    #     # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
    #     #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
    #     mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
    #     # $\sigma^2$
    #     var = gather(self.sigma2, t)
    #
    #     # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
    #     eps = torch.randn(xt.shape, device=self.device)
    #     # Sampling with reparameterization trick
    #     return mean + (var ** .5) * eps


DATADIR = '/data'

import torchvision.transforms.functional as TF

def batch_transform(batch):
    return {'input': [TF.to_tensor(img) for img in batch['image']], 'cond': batch['text']}


class LDMTest(unittest.TestCase):
    def setUp(self):
        self.config = OrderedDict({
            'uncond_scale': 7.5,
            'latent_scaling_factor': 0.18215,
            'n_steps': 1000,
            'beta_start': 0.0008,
            'beta_end': 0.012,
            'autoencoder': 'kl-f8',
            'cond_encoder': 'openai/clip-vit-large-patch14',
            'eps_model': {
                'channels': 320,
                'attention_levels': [0, 1, 2],
                'n_res_blocks': 2,
                'channel_multipliers': [1, 2, 4, 4],
                'n_heads': 8,
                'tf_layers': 1,
            }
        })
        self.model = LDM(config=self.config)

    def create_trainer(self, max_epochs=10):
        return pl.Trainer(accelerator='auto',
                          devices='auto',
                          strategy='ddp_spawn_find_unused_parameters_true',
                          accumulate_grad_batches=self.accum_grad_batches,
                          max_epochs=max_epochs,
                          precision=16,
                          enable_checkpointing=False)

    def test_fit_and_sample(self):
        trainer = self.create_trainer(max_epochs=1)
        trainer.fit(self.model, self.train_dataloader)
        #
        # self.model.eval()
        # with torch.no_grad():
        #     n = 16
        #     sample_size = (self.n_channels, self.img_size, self.img_size)
        #     samples = self.model.generate_sample(n=n, sample_size=sample_size)
        #     plt.imshow(make_grid(samples.view(-1, *sample_size), nrow=n//4, normalize=True).permute(1, 2, 0))
        #     plt.show()

if __name__ == '__main__':
    unittest.main()
