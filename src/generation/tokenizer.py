AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"

PAD = 0
END = 21
START = 22

aa_to_id = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}
id_to_aa = {i + 1: aa for i, aa in enumerate(AA_VOCAB)}

VOCAB_SIZE = 23  # 20 AA + PAD + END + START
MAX_LEN = 62     # START + 60 AA max + END

def encode(seq):
    """Encode peptide string to fixed-length token list."""
    seq = seq.strip().upper()
    tokens = [START]
    
    for aa in seq:
        if aa not in aa_to_id:
            continue  # Skip invalid amino acids
        tokens.append(aa_to_id[aa])
    
    tokens.append(END)
    tokens = tokens[:MAX_LEN]
    tokens += [PAD] * (MAX_LEN - len(tokens))
    return tokens

def decode(tokens):
    """Decode token list back to peptide string."""
    seq = []
    for t in tokens:
        if t == END:
            break
        if t in (PAD, START):
            continue
        if t in id_to_aa:
            seq.append(id_to_aa[t])
    return "".join(seq)