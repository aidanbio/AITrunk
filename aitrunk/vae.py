"""
Implementation of Variational AutoEncoder for MINST dataset
This code is based on the following code:
    hong_journey.log: https://velog.io/@hong_journey/VAEVariational-AutoEncoder-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0
    Nice explanation of VAE: 이활석님의 오코인코더의 모든것: https://www.youtube.com/watch?v=o_peo6U7IRM
"""
import copy
from collections import OrderedDict
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class VAELitModel(pl.LightningModule):
    class Encoder(nn.Module):
        """
        Encoder of VAE, estimate the latent variable z from input x, p(z|x) using reparameterization trick
        """
        def __init__(self, x_dim=28*28, h_dim=400, z_dim=10):
            super().__init__()
            # 1st hidden layer
            self.fc1 = nn.Sequential(
                nn.Linear(x_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            )

            # 2nd hidden layer
            self.fc2 = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            )
            # output layer for estimating mu, var
            self.mu = nn.Linear(h_dim, z_dim)
            self.logvar = nn.Linear(h_dim, z_dim)

        def forward(self, x):
            x = self.fc2(self.fc1(x))
            mu = F.relu(self.mu(x))
            logvar = F.relu(self.logvar(x))

            # Get latent variable z from mu, logvar, due to backpropagation,
            # we can't sample z from N(mu, var) directly, so we use reparameterization trick
            z = self.reparameterization(mu, logvar)
            return z, mu, logvar

        def reparameterization(self, mu, logvar):
            std = torch.exp(logvar/2)
            eps = torch.randn_like(std)
            return mu + eps * std

    class Decoder(nn.Module):
        """
        Decoder of VAE, estimate the reconstructed input x from latent variable z, p(x|z)
        """
        def __init__(self, x_dim=28*28, h_dim=400, z_dim=10):
            super().__init__()
            # 1st hidden layer
            self.fc1 = nn.Sequential(
                nn.Linear(z_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )

            # 2nd hidden layer
            self.fc2 = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            )

            # output layer
            self.fc3 = nn.Linear(h_dim, x_dim)

        # Reconstruct input x from latent variable z
        # In order to the MLE of p(x|z), p(x|z) is assumed to be Bernoulli distribution
        # and we use sigmoid function of the output layer, [0, 1].
        # The loss will be the binary cross entropy between the input x and the reconstructed x
        def forward(self, z):
            z = self.fc2(self.fc1(z))
            x_reconst = F.sigmoid(self.fc3(z))
            return x_reconst

    def __init__(self, x_dim=28*28, h_dim=400, z_dim=10):
        super().__init__()
        self.encoder = self.Encoder(x_dim, h_dim, z_dim)
        self.decoder = self.Decoder(x_dim, h_dim, z_dim)

    def configure_optimizers(self):
        return Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        return self._get_loss(batch, log_prefix='train')

    def validation_step(self, batch, batch_idx):
        return self._get_loss(batch, log_prefix='val')

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_reconst = self.decoder(z)
        return x_reconst, mu, logvar

    def generate_samples(self, n=1):
        z = torch.randn(n, self.encoder.mu.out_features)
        x_reconst = self.decoder(z)
        return x_reconst

    def _get_loss(self, batch, log_prefix='train'):
        x = batch[0]
        x = x.view(x.size(0), -1)
        x_reconst, mu, logvar = self(x)
        # Compute the reconstruction loss and the regularization term, KL divergence, which is q(x|z) || p(z)
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        loss = reconst_loss + kl_div
        self.log_dict(OrderedDict({f'{log_prefix}.loss': loss,
                                   f'{log_prefix}.reconst_loss': reconst_loss,
                                   f'{log_prefix}.kl_div': kl_div}),
                      prog_bar=True)
        return loss

class VAELitModelTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        datadir = '../data'
        batch_size = 128
        train = datasets.MNIST(root=datadir, train=True, transform=transform, download=True)
        test = datasets.MNIST(root=datadir, train=False, transform=transform, download=True)

        self.train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)
        img_size = 28
        n_channels = 1
        self.model = VAELitModel(x_dim=img_size*img_size*n_channels, h_dim=400, z_dim=10)

    def state_dict_equal(self, st1, st2):
        if st1.keys() != st2.keys():
            return False

        for k, v1 in st1.items():
            v2 = st2[k]
            if (not torch.equal(v1, v2)):
                return False
        return True

    def create_trainer(self, max_epochs=10):
        return pl.Trainer(accelerator='auto',
                          devices='auto',
                          strategy='ddp_spawn',
                          max_epochs=max_epochs,
                          enable_checkpointing=False)

    def test_fit_and_sample(self):
        old_sd = copy.deepcopy(self.model.state_dict())
        trainer = self.create_trainer(max_epochs=10)
        trainer.fit(self.model,
                    train_dataloaders=self.train_dataloader,
                    val_dataloaders=self.test_dataloader)
        new_sd = self.model.state_dict()
        self.assertFalse(self.state_dict_equal(old_sd, new_sd))

        self.model.eval()
        with torch.no_grad():
            samples = self.model.generate_samples(n=20)
            plt.imshow(make_grid(samples.view(-1, 1, 28, 28), nrow=4, normalize=True).permute(1, 2, 0))
            plt.show()

if __name__ == '__main__':
    unittest.main()
