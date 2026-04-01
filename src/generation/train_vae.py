import torch
from torch.utils.data import DataLoader
from dataset import PeptideDataset
from vae_model import VAE
import torch.nn as nn

def loss_fn(recon, x, mu, logvar, beta):
    recon_loss = nn.CrossEntropyLoss(ignore_index=0)(
        recon.reshape(-1, recon.size(-1)),
        x.reshape(-1)
    )

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + beta * kl_loss

dataset = PeptideDataset("data/processed/peptides.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # ✅ lower LR

for epoch in range(40):  # ✅ more epochs
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        # SHIFT
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]

        recon, mu, logvar = model(input_seq)

        beta = min(1.0, epoch / 10)

        loss = loss_fn(recon, target_seq, mu, logvar, beta)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch} Loss: {total_loss}")

torch.save(model.state_dict(), "models/vae_model.pt")