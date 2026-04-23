import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PeptideDataset
from vae_model import VAE, PAD

def loss_fn(recon, target, mu, logvar, beta):
    recon_loss = nn.CrossEntropyLoss(ignore_index=PAD)(
        recon.reshape(-1, recon.size(-1)),
        target.reshape(-1)
    )

    kl_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1
    )
    kl_loss = kl_per_sample.mean()

    return recon_loss, kl_loss, recon_loss + beta * kl_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PeptideDataset("data/processed/peptides.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 30
os.makedirs("models", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    beta = min(1.0, epoch / 10.0)

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        recon, target, mu, logvar = model(batch)
        recon_loss, kl_loss, loss = loss_fn(recon, target, mu, logvar, beta)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"beta={beta:.3f} | "
        f"loss={total_loss:.4f} | "
        f"recon={total_recon:.4f} | "
        f"kl={total_kl:.4f}"
    )

torch.save(model.state_dict(), "models/vae_model.pt")