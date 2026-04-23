import math
import torch
import torch.nn as nn

PAD = 0
VOCAB_SIZE = 23
EMBED_DIM = 128
LATENT_DIM = 64
MAX_LEN = 62

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD)
        self.pos_encoder = PositionalEncoding(EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc_mu = nn.Linear(EMBED_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(EMBED_DIM, LATENT_DIM)

        self.decoder_fc = nn.Linear(LATENT_DIM, EMBED_DIM)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.output_fc = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def _causal_mask(self, size, device):
        return torch.triu(
            torch.full((size, size), float("-inf"), device=device),
            diagonal=1
        )

    def encode(self, x):
        pad_mask = (x == PAD)
        x_emb = self.pos_encoder(self.embedding(x))
        h = self.encoder(x_emb, src_key_padding_mask=pad_mask)

        valid_mask = (~pad_mask).unsqueeze(-1)
        h_sum = (h * valid_mask).sum(dim=1)
        lengths = valid_mask.sum(dim=1).clamp(min=1)
        h_pooled = h_sum / lengths

        mu = self.fc_mu(h_pooled)
        logvar = self.fc_logvar(h_pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, decoder_input):
        batch_size, seq_len = decoder_input.size()
        tgt_pad_mask = (decoder_input == PAD)
        tgt_mask = self._causal_mask(seq_len, decoder_input.device)

        memory = self.decoder_fc(z).unsqueeze(1)
        memory = memory.repeat(1, seq_len, 1)

        tgt = self.pos_encoder(self.embedding(decoder_input))
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        return self.output_fc(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        decoder_input = x[:, :-1]
        target_output = x[:, 1:]

        recon = self.decode(z, decoder_input)
        return recon, target_output, mu, logvar