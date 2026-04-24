import torch
import torch.nn.functional as F
import argparse
from vae_model import VAE, LATENT_DIM
from tokenizer import decode, START, END, PAD, AA_VOCAB
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = "models/vae_model.pt"
model = VAE().to(device)


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Train first: {path}")
    state = torch.load(path, map_location=device)
    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"checkpoint warning | missing={msg.missing_keys} unexpected={msg.unexpected_keys}")
    model.eval()


load_model(model_path)

def sample_next_token(logits, prev_tokens, temperature=0.8):
    """Sample with repetition penalty and mask specials."""
    logits = logits.clone()
    
    # Mask invalid tokens
    logits[PAD] = float('-inf')
    logits[START] = float('-inf')
    
    # Repetition penalty
    for token in prev_tokens[-4:]:
        logits[token] *= 0.8
    
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

@torch.no_grad()
def generate(n=10, max_new_tokens=60):
    results = []
    valid_aa = set(AA_VOCAB)
    
    for i in range(n):
        z = torch.randn(1, LATENT_DIM, device=device)
        tokens = [START]
        
        for _ in range(max_new_tokens):
            input_seq = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decode(z, input_seq)[0, -1]
            
            next_token = sample_next_token(logits, tokens)
            tokens.append(next_token)
            
            if next_token == END:
                break
        
        peptide = decode(tokens)
        if peptide and set(peptide).issubset(valid_aa):
            results.append(peptide)
        print(f"{i+1:2d}: {peptide}")

    unique_ratio = len(set(results)) / max(1, len(results))
    duplicate_rate = 1.0 - unique_ratio if results else 0.0
    avg_len = sum(map(len, results)) / max(1, len(results))
    print(
        f"sanity | valid={len(results)}/{n} | unique_ratio={unique_ratio:.4f} "
        f"| duplicate_rate={duplicate_rate:.4f} | avg_len={avg_len:.2f}"
    )
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--model", default=model_path)
    args = parser.parse_args()
    load_model(args.model)
    generate(args.n, args.max_new_tokens)