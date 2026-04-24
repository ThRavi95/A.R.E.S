import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import PeptideDataset
from vae_model import VAE, PAD
from tokenizer import VOCAB_SIZE

def compute_loss(recon, target, mu, logvar, beta, free_bits=0.02):
    recon_loss = nn.CrossEntropyLoss(ignore_index=PAD, reduction='mean')(
        recon.reshape(-1, VOCAB_SIZE), target.reshape(-1)
    )
    
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_raw = kl_per_dim.sum(dim=1).mean()
    kl_loss = torch.clamp(kl_per_dim.mean(dim=0), min=free_bits).sum()
    
    vae_loss = recon_loss + beta * kl_loss
    return recon_loss, kl_raw, kl_loss, vae_loss


@torch.no_grad()
def evaluate(model, loader, device, beta):
    model.eval()
    total_recon, total_kl_raw, total_kl_loss, total_loss, total_mu_abs = 0.0, 0.0, 0.0, 0.0, 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        recon, target, mu, logvar = model(batch)
        recon_loss, kl_raw, kl_loss, loss = compute_loss(recon, target, mu, logvar, beta)

        total_recon += recon_loss.item()
        total_kl_raw += kl_raw.item()
        total_kl_loss += kl_loss.item()
        total_loss += loss.item()
        total_mu_abs += mu.abs().mean().item()
        num_batches += 1

    return {
        "recon": total_recon / num_batches,
        "kl_raw": total_kl_raw / num_batches,
        "kl_loss": total_kl_loss / num_batches,
        "loss": total_loss / num_batches,
        "mu_abs": total_mu_abs / num_batches,
    }


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = PeptideDataset(args.data)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs("models", exist_ok=True)

    best_val = float("inf")
    warmup_epochs = max(1, args.warmup_epochs)

    for epoch in range(args.epochs):
        model.train()
        total_recon, total_kl_raw, total_kl_loss, total_loss, total_mu_abs = 0.0, 0.0, 0.0, 0.0, 0.0
        num_batches = 0

        beta = min(args.max_beta, args.max_beta * (epoch + 1) / warmup_epochs)

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            recon, target, mu, logvar = model(batch)
            recon_loss, kl_raw, kl_loss, loss = compute_loss(recon, target, mu, logvar, beta, free_bits=args.free_bits)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl_raw += kl_raw.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()
            total_mu_abs += mu.abs().mean().item()
            num_batches += 1

        train_stats = {
            "recon": total_recon / num_batches,
            "kl_raw": total_kl_raw / num_batches,
            "kl_loss": total_kl_loss / num_batches,
            "loss": total_loss / num_batches,
            "mu_abs": total_mu_abs / num_batches,
        }
        val_stats = evaluate(model, val_loader, device, beta)

        print(
            f"Epoch {epoch+1:2d}/{args.epochs} | beta={beta:.3f} | "
            f"train recon={train_stats['recon']:.4f} kl_raw={train_stats['kl_raw']:.4f} "
            f"kl_loss={train_stats['kl_loss']:.4f} | val recon={val_stats['recon']:.4f} "
            f"kl_raw={val_stats['kl_raw']:.4f} kl_loss={val_stats['kl_loss']:.4f} "
            f"| train |mu|={train_stats['mu_abs']:.4f} val |mu|={val_stats['mu_abs']:.4f}"
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(model.state_dict(), args.out)

    print(f"Best model saved to {args.out} (val_loss={best_val:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/peptides.csv")
    parser.add_argument("--out", default="models/vae_model.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-beta", type=float, default=0.4)
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--free-bits", type=float, default=0.02)
    args = parser.parse_args()
    train(args)