# Fichier: src/dataset.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import pandas as pd
from PIL import Image
import io
import os

class CadCodeDataset(Dataset):
    """
    Dataset PyTorch pour les paires Image/Code CadQuery, lisant les données
    depuis des fichiers Parquet pré-téléchargés.
    """
    def __init__(self, parquet_files, tokenizer_path, fraction=1.0):
        """
        Args:
            parquet_files (list): Liste des chemins vers les fichiers Parquet.
            tokenizer_path (str): Chemin vers le fichier JSON du tokenizer.
            fraction (float): Fraction du dataset à utiliser (entre 0.0 et 1.0).
        """
        # Charger le vocabulaire du tokenizer
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
        self.char_to_int = tokenizer_data['char_to_int']
        self.int_to_char = tokenizer_data['int_to_char']
        
        # Définir les IDs des tokens spéciaux pour un accès facile
        self.pad_token_id = self.char_to_int['<PAD>']
        self.sos_token_id = self.char_to_int['<SOS>']
        self.eos_token_id = self.char_to_int['<EOS>']
        self.unk_token_id = self.char_to_int['<UNK>']
        
        self.vocab_size = len(self.char_to_int)

        # Charger les données depuis les fichiers Parquet
        print("Chargement des données depuis les fichiers Parquet...")
        df_list = [pd.read_parquet(f) for f in parquet_files]
        self.dataframe = pd.concat(df_list, ignore_index=True)
        
        # Utiliser une fraction du dataset si spécifié pour des tests rapides
        if fraction < 1.0:
            self.dataframe = self.dataframe.sample(frac=fraction, random_state=42).reset_index(drop=True)
        
        print(f"{len(self.dataframe)} échantillons chargés dans le dataset.")

        # Transformations standard pour les images (pour modèles pré-entraînés sur ImageNet)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # 1. Traitement de l'image
        image_data = row['image'] # C'est un dictionnaire {'bytes': ...}
        
        # On ouvre l'image à partir des données binaires contenues dans le dictionnaire
        image = Image.open(io.BytesIO(image_data['bytes'])).convert("RGB")
        
        image_tensor = self.transform(image)
        
        # 2. Traitement du code
        code = row['cadquery']
        
        # Tokenizer le code en ajoutant les tokens de début et de fin
        tokenized_code = [self.sos_token_id]
        tokenized_code.extend([self.char_to_int.get(char, self.unk_token_id) for char in code])
        tokenized_code.append(self.eos_token_id)
        
        code_tensor = torch.LongTensor(tokenized_code)
        
        return image_tensor, code_tensor

# --- Bloc de test pour vérifier que le Dataset fonctionne ---
if __name__ == '__main__':
    TOKENIZER_FILE = "tokenizer_vocab.json"
    PARQUET_FILES = ['train_data_part1.parquet', 'train_data_part2.parquet']

    if not os.path.exists(TOKENIZER_FILE) or not all(os.path.exists(f) for f in PARQUET_FILES):
        print("Erreur : Un ou plusieurs fichiers requis sont manquants.")
        print("Veuillez d'abord lancer le script 'explore_and_tokenize_manual.py'.")
    else:
        print("Test de la classe CadCodeDataset...")
        # On teste avec une petite fraction pour aller vite
        dataset = CadCodeDataset(PARQUET_FILES, TOKENIZER_FILE, fraction=0.001)
        
        img_tensor, code_tensor = dataset[0]
        
        print("\nTest réussi !")
        print(f"Shape du tenseur image : {img_tensor.shape}")
        print(f"Shape du tenseur code : {code_tensor.shape}")
        print(f"Taille du vocabulaire : {dataset.vocab_size}")