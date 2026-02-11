from pytorch_lightning import Trainer
import torch
import torch.nn as nn
import pytorch_lightning as pl
from datasets.dataset import AD_Dataset
from torch.utils.data import DataLoader

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
        self.fc1 = nn.Linear(64 * 16 * 32, latent_dim)  # adjust depending on input size
        self.fc2 = nn.Linear(latent_dim, 64 * 16 * 32)

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
        print(enc.shape)
        enc_flat = self.flatten(enc)
        print(enc_flat.shape)
        latent = self.fc1(enc_flat)
        print(latent.shape)
        dec_flat = self.fc2(latent)
        dec = dec_flat.view(x.size(0), *self.enc_shape)
        out = self.decoder(dec)
        return out



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


def compute_anomaly_score(model, volume):
    """
    volume: [1, C, D, H, W]
    Returns: reconstruction error per voxel
    """
    model.eval()
    with torch.no_grad():
        recon = model(volume)
        # voxel-wise MSE
        score = (recon - volume).pow(2)
        volume_score = score.mean().item()
    return score, volume_score

lit_model = LitAutoencoder3D(in_channels=1, lr=1e-3)

trainer = Trainer(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
)

# Construct Dataset & DataLoader
train_dset = AD_Dataset(name="mri", img_size=128, train=True, data_dir="/projects/prjs1633/anomaly_detection/SHOMRI/")
train_dset = train_dset.get_dset()
print('TrainSet Image Number:', len(train_dset))
eval_dset = AD_Dataset(name="mri", img_size=128, train=False, data_dir="/projects/prjs1633/anomaly_detection/SHOMRI/")
eval_dset = eval_dset.get_dset()
print('EvalSet Image Number:', len(eval_dset))

loader_dict = {}
dset_dict = {'train': train_dset, 'eval': eval_dset}


train_loader = DataLoader(train_dset, batch_size=8, shuffle=True, num_workers=4)
eval_loader = DataLoader(eval_dset, batch_size=8, shuffle=True, num_workers=4)


trainer.fit(lit_model, train_loader)



print(compute_anomaly_score(lit_model, loader_dict['eval']))