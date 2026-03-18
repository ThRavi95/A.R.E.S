import torch
import torch.nn.functional as F
from vae_model import VAE
from tokenizer import decode

model = VAE()
model.load_state_dict(torch.load("models/vae_model.pt"))
model.eval()

def sample_from_logits(logits, prev_tokens):
    temperature = 0.8

    probs = F.softmax(logits / temperature, dim=-1)

    # Penalize repetition
    for t in prev_tokens[-5:]:
        probs[t] *= 0.7

    probs = probs / probs.sum()

    return torch.multinomial(probs, 1).item()

def generate(n=10):
    results = []

    for _ in range(n):
        z = torch.randn(1, 64)

        tokens = [21]  # START token

        for _ in range(60):
            input_seq = torch.tensor([tokens], dtype=torch.long)

            out = model.decode(z, input_seq)
            next_token_logits = out[0, -1]

            # next_token = sample_from_logits(next_token_logits)
            next_token = sample_from_logits(next_token_logits, tokens)
            if len(tokens) > 10 and tokens[-5:] == [tokens[-1]] * 5:
                break
            if next_token == 0:  # PAD → stop
                break

            tokens.append(next_token)

        seq = decode(tokens)
        results.append(seq)

    return results

peptides = generate(10)

for p in peptides:
    print(p)