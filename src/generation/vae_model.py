import torch
import math
import torch.nn as nn

VOCAB_SIZE = 22
EMBED_DIM = 128
LATENT_DIM = 64
MAX_LEN = 60

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.pos_encoder = PositionalEncoding(EMBED_DIM)
        # ✅ FIX: batch_first=True + syntax fix
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=8,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc_mu = nn.Linear(EMBED_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(EMBED_DIM, LATENT_DIM)

        self.decoder_fc = nn.Linear(LATENT_DIM, EMBED_DIM)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM,
            nhead=8,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.output_fc = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def encode(self, x):
        x = self.embedding(x)  # (batch, seq, dim)
        x = self.pos_encoder(x)
        h = self.encoder(x)

        h = h.mean(dim=1)  # ✅ FIX: correct dimension

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target):
        batch_size, seq_len = target.size()

        z = self.decoder_fc(z)  # (batch, embed)

        # ✅ FIX: correct expansion
        z = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq, embed)

        tgt = self.embedding(target)  # (batch, seq, embed)
        tgt = self.pos_encoder(tgt)
        out = self.decoder(tgt, z)

        return self.output_fc(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x)
        return recon, mu, logvar