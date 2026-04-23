import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PeptideDataset
from vae_model import VAE, PAD
from tokenizer import VOCAB_SIZE

def compute_loss(recon, target, mu, logvar, beta):
    recon_loss = nn.CrossEntropyLoss(ignore_index=PAD, reduction='mean')(
        recon.reshape(-1, VOCAB_SIZE), target.reshape(-1)
    )
    
    # KL divergence per sample, then mean
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    vae_loss = recon_loss + beta * kl_loss
    return recon_loss, kl_loss, vae_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data
dataset = PeptideDataset("data/processed/peptides.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# Model & optimizer
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
os.makedirs("models", exist_ok=True)

# Training loop
num_epochs = 75
for epoch in range(num_epochs):
    model.train()
    total_recon, total_kl, total_loss = 0.0, 0.0, 0.0
    num_batches = 0
    
    beta = min(1.0, epoch / 10.0)
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        recon, target, mu, logvar = model(batch)
        recon_loss, kl_loss, loss = compute_loss(recon, target, mu, logvar, beta)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_loss += loss.item()
        num_batches += 1
    
    avg_recon = total_recon / num_batches
    avg_kl = total_kl / num_batches
    avg_loss = total_loss / num_batches
    
    print(f"Epoch {epoch+1:2d}/{num_epochs} | "
          f"beta={beta:.2f} | "
          f"recon={avg_recon:.4f} | "
          f"kl={avg_kl:.4f} | "
          f"loss={avg_loss:.4f}")

torch.save(model.state_dict(), "models/vae_model.pt")
print("Model saved to models/vae_model.pt")