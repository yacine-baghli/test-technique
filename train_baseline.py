# Fichier: train_baseline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import CadCodeDataset
from src.model_baseline import EncoderDecoderBaseline

# --- Fonction utilitaire pour le batching ---
def collate_fn(batch):
    """
    Fonction pour rassembler des données de longueurs variables en un seul batch.
    Elle ajoute du "padding" pour que tous les codes aient la même longueur.
    """
    images, codes = zip(*batch)
    images = torch.stack(images, 0)
    
    # Trouver la longueur max du code dans ce batch
    max_len = max(len(c) for c in codes)
    
    # "Padder" chaque code avec le token PAD
    padded_codes = torch.zeros(len(codes), max_len).long()
    for i, c in enumerate(codes):
        end = len(c)
        padded_codes[i, :end] = c[:end]
        
    return images, padded_codes

# --- Paramètres ---
TOKENIZER_PATH = 'tokenizer_vocab.json'
PARQUET_FILES = ['train_data_part1.parquet', 'train_data_part2.parquet']
# IMPORTANT : On s'entraîne sur un tout petit sous-ensemble pour aller vite
DATA_FRACTION = 0.05 # 5%
BATCH_SIZE = 32
EMBED_SIZE = 256
HIDDEN_SIZE = 512
EPOCHS = 10

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    # 1. Charger les données
    dataset = CadCodeDataset(
        parquet_files=PARQUET_FILES,
        tokenizer_path=TOKENIZER_PATH,
        fraction=DATA_FRACTION
    )
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn # On utilise notre fonction de padding
    )
    
    # 2. Initialiser le modèle
    model = EncoderDecoderBaseline(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=dataset.vocab_size
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Boucle d'entraînement
    print("\n--- Démarrage de l'entraînement de la Baseline ---")
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (images, codes) in enumerate(data_loader):
            images, codes = images.to(device), codes.to(device)
            
            optimizer.zero_grad()
            
            # Préparer les entrées et sorties du décodeur
            # Le modèle doit prédire le code à partir du token précédent
            targets = codes[:, 1:]
            decoder_inputs = codes[:, :-1]
            
            outputs = model(images, decoder_inputs)
            
            # La sortie est [batch, seq_len, vocab_size], la loss a besoin de [batch*seq_len, vocab_size]
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')
        
        print(f'---> Epoch [{epoch+1}/{EPOCHS}] - Loss moyenne: {total_loss/len(data_loader):.4f}')
        
    # Sauvegarder le modèle de baseline
    torch.save(model.state_dict(), 'baseline_model.pth')
    print("\nModèle de baseline sauvegardé dans 'baseline_model.pth'")

if __name__ == '__main__':
    main()