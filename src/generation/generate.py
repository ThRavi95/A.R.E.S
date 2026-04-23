import torch
import torch.nn.functional as F
from vae_model import VAE
from tokenizer import decode, START, END, PAD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE().to(device)
model.load_state_dict(torch.load("models/vae_model.pt", map_location=device))
model.eval()

def sample_from_logits(logits, prev_tokens, temperature=0.8):
    logits = logits.clone()

    logits[PAD] = -1e9
    logits[START] = -1e9

    probs = F.softmax(logits / temperature, dim=-1)

    for t in prev_tokens[-5:]:
        probs[t] *= 0.7

    probs = probs / probs.sum()
    return torch.multinomial(probs, 1).item()

@torch.no_grad()
def generate(n=10, max_len=60):
    results = []

    for _ in range(n):
        z = torch.randn(1, 64, device=device)
        tokens = [START]

        for _ in range(max_len):
            input_seq = torch.tensor([tokens], dtype=torch.long, device=device)
            out = model.decode(z, input_seq)
            next_token = sample_from_logits(out[0, -1], tokens)

            tokens.append(next_token)

            if next_token == END:
                break

            if len(tokens) >= 6 and len(set(tokens[-5:])) == 1:
                break

        results.append(decode(tokens))

    return results

if __name__ == "__main__":
    peptides = generate(10)
    for p in peptides:
        print(p)