"""
Pytorch Lightning implementation of Denoising Diffusion Probabilistic Models (DDPM)
Paper: https://arxiv.org/abs/2006.11239
This implementation is based on labml.ai's implementations:
https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/ddpm
"""
import unittest
from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from aitrunk.diffusion.ddpm.utils import gather
from aitrunk.diffusion.ddpm.unet import UNet

class DDPLitModel(pl.LightningModule):
    def __init__(self, eps_model: nn.Module, n_steps: int):
        """
        :param eps_model: ${\epsilon_\theta}(x_t, t)$ model that predicts noised samples at time $t$
        :param n_steps: the number of time steps $T$
        """
        super().__init__()

        self.eps_model = eps_model
        self.n_steps = n_steps
        # Create β1, β2, …, βT linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # σ^2 = β
        self.sigma2 = self.beta

    def training_step(self, batch, batch_idx):
        return self.loss_and_log(batch, log_prefix='train')

    def validation_step(self, batch, batch_idx):
        return self.loss_and_log(batch, log_prefix='val')

    def configure_callbacks(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer

    def loss_and_log(self, batch, log_prefix='train'):
        imgs, _ = batch
        loss = self.get_loss(imgs)
        self.log(f'{log_prefix}.loss', loss, prog_bar=True)
        return loss

    def get_loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Compute loss between source noise from N(0, I) and predicted noise from x0 to xt
        :param x0: source data
        :param noise: source noise
        """
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, eps=noise)
        # Get ${\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t)

        # MSE loss
        return F.mse_loss(noise, eps_theta)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        Sample xt from $q(x_t|x_0)$ with reparameterization trick
        """
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Get encoder transition $q(x_t|x_0)$ distribution
        """
        self.alpha_bar = self.alpha_bar.to(self.device)
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def generate_sample(self, n=10, sample_size=(1, 32, 32)):
        """
        Algorithm 2: Sampling from xT to x0
        """
        # Random noisy sample $x_T$
        x = torch.randn((n, *sample_size), device=self.device)
        # Reverse process to remove noise from $x_T$ to $x_0$
        for t in reversed(range(self.n_steps)):
            x = self.p_sample(x, x.new_full((n,), t, dtype=torch.long))
            print(f'Reversed to remove noise at {t}')
        return x

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        Sample from ${p_\theta}(x_{t-1}|x_t)$
        :param xt: noised sample $x_t$
        :param t: time step $t$
        """
        self.alpha = self.alpha.to(self.device)
        self.alpha_bar = self.alpha_bar.to(self.device)
        self.sigma2 = self.sigma2.to(self.device)

        # ${\epsilon_\theta}(x_t, t)$
        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=self.device)
        # Sampling with reparameterization trick
        return mean + (var ** .5) * eps


DATADIR = '/data'


class DDLitModelTest(unittest.TestCase):
    def setUp(self):
        self.img_size = 32
        self.n_channels = 1
        transform = transform=transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])
        train_ds = datasets.MNIST(root=DATADIR, train=True, transform=transform, download=True)
        val_ds = datasets.MNIST(root=DATADIR, train=False, transform=transform, download=True)

        self.batch_size = 256
        self.accum_grad_batches = 4
        self.train_dataloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        self.n_steps = 1000
        self.model = self.create_model()

    def create_model(self):
        eps_model = UNet(image_channels=self.n_channels)
        return DDPLitModel(eps_model, self.n_steps)

    def create_trainer(self, max_epochs=10):
        return pl.Trainer(accelerator='auto',
                          devices='auto',
                          strategy='ddp_spawn_find_unused_parameters_true',
                          accumulate_grad_batches=self.accum_grad_batches,
                          max_epochs=max_epochs,
                          enable_checkpointing=False)

    def test_fit_and_sample(self):
        trainer = self.create_trainer(max_epochs=10)
        trainer.fit(self.model, self.train_dataloader, self.val_dataloader)

        self.model.eval()
        with torch.no_grad():
            n = 16
            sample_size = (self.n_channels, self.img_size, self.img_size)
            samples = self.model.generate_sample(n=n, sample_size=sample_size)
            plt.imshow(make_grid(samples.view(-1, *sample_size), nrow=n//4, normalize=True).permute(1, 2, 0))
            plt.show()

if __name__ == '__main__':
    unittest.main()
