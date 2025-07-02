# Fichier: train_enhanced.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import CadCodeDataset
# On importe le nouveau modèle amélioré
from src.model_enhanced import EncoderDecoderEnhanced
from train_baseline import collate_fn # On réutilise la même fonction

# --- Paramètres ---
TOKENIZER_PATH = 'tokenizer_vocab.json'
PARQUET_FILES = ['train_data_part1.parquet', 'train_data_part2.parquet']
DATA_FRACTION = 0.05 # 5%
BATCH_SIZE = 32
# Le 'embed_size' du Transformer est couplé au 'hidden_size' dans notre modèle
HIDDEN_SIZE = 512 
EPOCHS = 15 # Un peu plus long car le modèle est plus complexe

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    dataset = CadCodeDataset(
        parquet_files=PARQUET_FILES,
        tokenizer_path=TOKENIZER_PATH,
        fraction=DATA_FRACTION
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Initialiser le modèle amélioré
    model = EncoderDecoderEnhanced(
        embed_size=HIDDEN_SIZE, # embed_size = hidden_size
        hidden_size=HIDDEN_SIZE,
        vocab_size=dataset.vocab_size
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Un learning rate un peu plus bas

    print("\n--- Démarrage de l'entraînement du Modèle Amélioré (Transformer) ---")
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (images, codes) in enumerate(data_loader):
            images, codes = images.to(device), codes.to(device)
            optimizer.zero_grad()
            targets = codes[:, 1:]
            decoder_inputs = codes[:, :-1]
            outputs = model(images, decoder_inputs)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')
        
        print(f'---> Epoch [{epoch+1}/{EPOCHS}] - Loss moyenne: {total_loss/len(data_loader):.4f}')
        
    torch.save(model.state_dict(), 'enhanced_model.pth')
    print("\nModèle amélioré sauvegardé dans 'enhanced_model.pth'")

if __name__ == '__main__':
    main()