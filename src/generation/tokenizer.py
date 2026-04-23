AA_vocab = "ACDEFGHIKLMNPQRSTVWY"

aa_to_id = {aa: i+1 for i, aa in enumerate(AA_vocab)}
id_to_aa = {i+1: aa for i, aa in enumerate(AA_vocab)}

PAD = 0
END = 21
START = 22

VOCAB_SIZE = 23  # 20 AA + PAD + START
MAX_LEN = 62

def encode(seq):
    seq = seq.strip().upper()
    tokens = [START]

    for aa in seq:
        if aa not in aa_to_id:
            raise ValueError(f"Invalid amino acid: {aa}")
        tokens.append(aa_to_id[aa])

    tokens.append(END)
    tokens = tokens[:MAX_LEN]
    tokens += [PAD] * (MAX_LEN - len(tokens))
    return tokens

def decode(tokens):
    seq = []
    for t in tokens:
        if t == END:
            break
        if t in (PAD, START):
            continue
        if t in id_to_aa:
            seq.append(id_to_aa[t])
    return "".join(seq)