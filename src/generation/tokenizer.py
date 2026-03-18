AA_vocab = "ACDEFGHIKLMNPQRSTVWY"

aa_to_id = {aa: i+1 for i, aa in enumerate(AA_vocab)}
id_to_aa = {i+1: aa for i, aa in enumerate(AA_vocab)}

PAD = 0
START = 21

VOCAB_SIZE = 22  # 20 AA + PAD + START
MAX_LEN = 60

def encode(seq):
    tokens = [START] + [aa_to_id[a] for a in seq]
    tokens = tokens[:MAX_LEN]
    tokens += [PAD] * (MAX_LEN - len(tokens))
    return tokens

def decode(tokens):
    seq = ""
    for t in tokens:
        if t in [PAD, START]:
            continue
        seq += id_to_aa.get(t, "")
    return seq