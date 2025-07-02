# Fichier: explore_and_tokenize.py (version corrigée avec streaming)
import json
import os
from datasets import load_dataset
from itertools import islice

# --- Paramètres ---
TOKENIZER_FILE = "tokenizer_vocab.json"
# En mode streaming, on définit le nombre d'exemples qu'on veut utiliser
NUM_SAMPLES_FOR_VOCAB = 10000 

def build_and_save_tokenizer():
    """
    Charge un échantillon du dataset en mode streaming, construit un tokenizer
    et le sauvegarde dans un fichier JSON.
    """
    print("--- Phase 1 : Exploration et Création du Tokenizer (Mode Streaming) ---")
    
    # 1. Charger le dataset en mode streaming
    # Note : num_proc n'est pas compatible avec le streaming, on le retire.
    print(f"Chargement du dataset en mode streaming...")
    try:
        streaming_dataset = load_dataset(
            "CADCODER/GenCAD-Code",
            split='train',
            streaming=True 
        )
    except Exception as e:
        print(f"Erreur lors du chargement du dataset: {e}")
        return
    
    print("Dataset en streaming initialisé.")

    # 2. Construire le vocabulaire à partir d'un sous-ensemble
    print(f"Construction du vocabulaire à partir des {NUM_SAMPLES_FOR_VOCAB} premiers exemples...")
    
    # On prend les N premiers exemples du flux de données
    dataset_sample = islice(streaming_dataset, NUM_SAMPLES_FOR_VOCAB)
    
    corpus = [sample['code'] for sample in dataset_sample]
    all_chars = set()
    for code in corpus:
        all_chars.update(list(code))
    
    sorted_chars = sorted(list(all_chars))
    
    # 3. Créer les dictionnaires de mapping
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    all_tokens = special_tokens + sorted_chars
    
    char_to_int = {token: i for i, token in enumerate(all_tokens)}
    int_to_char = {i: token for i, token in enumerate(all_tokens)}
    
    tokenizer_data = {'char_to_int': char_to_int, 'int_to_char': int_to_char}
    
    # 4. Sauvegarder le tokenizer
    with open(TOKENIZER_FILE, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        
    print(f"\nTokenizer sauvegardé dans '{TOKENIZER_FILE}' avec {len(all_tokens)} tokens.")
    print("--- Phase 1 terminée avec succès ! ---")

if __name__ == '__main__':
    build_and_save_tokenizer()