import torch
import torch.nn.functional as F
from vae_model import VAE
from tokenizer import decode, START, END, PAD
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = "models/vae_model.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Train first: {model_path}")

model = VAE().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

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
    
    for i in range(n):
        z = torch.randn(1, 64, device=device)
        tokens = [START]
        
        for _ in range(max_new_tokens):
            input_seq = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decode(z, input_seq)[0, -1]
            
            next_token = sample_next_token(logits, tokens)
            tokens.append(next_token)
            
            if next_token == END:
                break
            if len(set(tokens[-5:])) == 1:  # Repetition
                break
        
        peptide = decode(tokens)
        results.append(peptide)
        print(f"{i+1:2d}: {peptide}")
    
    return results

if __name__ == "__main__":
    generate(10)