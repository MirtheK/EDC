import torch
import torch.nn as nn
import pytorch_lightning as pl

class Autoencoder3D(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, stride=2, padding=1),  # [B,16,D/2,H/2,W/2]
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),           # [B,32,D/4,H/4,W/4]
            nn.ReLU(True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),           # [B,64,D/8,H/8,W/8]
            nn.ReLU(True),
        )

        # Bottleneck
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 32 * 32, latent_dim)  # adjust depending on input size
        self.fc2 = nn.Linear(latent_dim, 64 * 16 * 32 * 32)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # for normalized intensity
        )

    def forward(self, x):
        enc = self.encoder(x)
        batch_size = x.size(0)
        enc_flat = self.flatten(enc)
        latent = self.fc1(enc_flat)
        dec_flat = self.fc2(latent)
        dec = dec_flat.view_as(enc)
        out = self.dec



class LitAutoencoder3D(pl.LightningModule):
    def __init__(self, in_channels=1, lr=1e-3):
        super().__init__()
        self.model = Autoencoder3D(in_channels)
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)