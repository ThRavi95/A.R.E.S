import math
import torch
import torch.nn as nn
from tokenizer import PAD, VOCAB_SIZE, MAX_LEN, START

EMBED_DIM = 128
LATENT_DIM = 64

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD)
        self.pos_encoder = PositionalEncoding(EMBED_DIM)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=8, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc_mu = nn.Linear(EMBED_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(EMBED_DIM, LATENT_DIM)
        
        self.decoder_fc = nn.Linear(LATENT_DIM, EMBED_DIM)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM, nhead=8, batch_first=True, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        self.output_fc = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def _causal_mask(self, size, device):
        mask = torch.triu(torch.full((size, size), float('-inf'), device=device), diagonal=1)
        return mask

    def encode(self, x):
        """Encode with padding mask and masked mean pooling."""
        src_key_padding_mask = (x == PAD)
        x_emb = self.pos_encoder(self.embedding(x))
        h = self.encoder(x_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Masked mean pooling
        valid_mask = (~src_key_padding_mask).float().unsqueeze(-1)
        h_masked = h * valid_mask
        h_sum = h_masked.sum(dim=1)
        seq_len = valid_mask.sum(dim=1)
        h_pooled = h_sum / seq_len.clamp(min=1.0)
        
        mu = self.fc_mu(h_pooled)
        logvar = self.fc_logvar(h_pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, decoder_input):
        """Decode with causal mask and padding mask."""
        batch_size, seq_len = decoder_input.shape
        tgt_key_padding_mask = (decoder_input == PAD)
        tgt_mask = self._causal_mask(seq_len, decoder_input.device)
        
        # Memory: repeat z across sequence positions
        memory = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        
        tgt = self.pos_encoder(self.embedding(decoder_input))
        output = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.output_fc(output)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Shifted targets for teacher forcing
        decoder_input = x[:, :-1]  # All but last token
        recon = self.decode(z, decoder_input)
        target = x[:, 1:]          # All but first token
        
        return recon, target, mu, logvar