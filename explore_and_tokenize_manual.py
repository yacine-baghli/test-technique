# Fichier: explore_and_tokenize_manual.py (version finale et corrigée)
import pandas as pd
import json
from tqdm import tqdm

TOKENIZER_FILE = "tokenizer_vocab.json"

print("--- Phase 1 (Manuelle) : Création du Tokenizer à partir des fichiers Parquet ---")

# 1. Lire les fichiers Parquet téléchargés
print("Lecture des fichiers Parquet...")
df1 = pd.read_parquet('train_data_part1.parquet')
df2 = pd.read_parquet('train_data_part2.parquet')
full_df = pd.concat([df1, df2], ignore_index=True)
print(f"Total de {len(full_df)} exemples chargés.")

# 2. Construire le vocabulaire à partir de la bonne colonne
print("Construction du vocabulaire...")

# --- LA CORRECTION EST ICI ---
corpus = full_df['cadquery']

all_chars = set()
for code in tqdm(corpus, desc="Analyse du corpus"):
    if isinstance(code, str):
        all_chars.update(list(code))

sorted_chars = sorted(list(all_chars))

# 3. Créer et sauvegarder le tokenizer
special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
all_tokens = special_tokens + sorted_chars

char_to_int = {token: i for i, token in enumerate(all_tokens)}
int_to_char = {i: token for i, token in enumerate(all_tokens)}

tokenizer_data = {'char_to_int': char_to_int, 'int_to_char': int_to_char}

with open(TOKENIZER_FILE, 'w', encoding='utf-8') as f:
    json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

print(f"\nTokenizer sauvegardé dans '{TOKENIZER_FILE}' avec {len(all_tokens)} tokens.")
print("--- Phase 1 (Manuelle) terminée avec succès ! ---")