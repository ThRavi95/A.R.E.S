import pandas as pd
from Bio import SeqIO
import torch
import os

def process_pipeline():
    print("🚀 Starting A.R.E.S Data Pipeline...")

    # 1. Define paths
    dbaasp_path = "data/raw/dbaasp.csv"
    apd3_path = "data/raw/apd3.fasta"
    
    sequences = []

    # 2. Load DBAASP (assuming the sequence column is named 'Sequence')
    if os.path.exists(dbaasp_path):
        print("Loading DBAASP...")
        df_dbaasp = pd.read_csv(dbaasp_path)
        # Adjust 'Sequence' if the column name in CSV is different (e.g., 'sequence', 'Peptide')
        if 'Sequence' in df_dbaasp.columns:
            sequences.extend(df_dbaasp['Sequence'].dropna().tolist())
        else:
            print("⚠️ Could not find 'Sequence' column in DBAASP.")

    # 3. Load APD3
    if os.path.exists(apd3_path):
        print("Loading APD3...")
        for record in SeqIO.parse(apd3_path, "fasta"):
            sequences.append(str(record.seq))

    # 4. Filter & Clean
    print(f"Total raw sequences: {len(sequences)}")
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
    
    # Remove duplicates, valid AA only, length between 10 and 60
    cleaned_sequences = []
    for seq in set(sequences):
        if all(c in VALID_AA for c in seq) and (10 <= len(seq) <= 60):
            cleaned_sequences.append(seq)
            
    print(f"Cleaned and filtered sequences: {len(cleaned_sequences)}")

    # 5. Save to CSV
    df_clean = pd.DataFrame({"sequence": cleaned_sequences})
    df_clean.to_csv("data/processed/peptides.csv", index=False)
    print("✅ Saved cleaned dataset to data/processed/peptides.csv")

    # 6. Encode for AI
    print("Encoding sequences for PyTorch...")
    AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_id = {aa: i+1 for i, aa in enumerate(AA_vocab)}
    
    encoded = [[aa_to_id[a] for a in seq] for seq in cleaned_sequences]
    
    # 7. Save Tensor
    torch.save(encoded, "data/processed/encoded_sequences.pt")
    print("✅ Saved encoded tensors to data/processed/encoded_sequences.pt")
    print("🎉 Phase 1 Complete!")

if __name__ == "__main__":
    process_pipeline()