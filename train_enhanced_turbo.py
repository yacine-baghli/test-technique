import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# On importe les outils pour l'entraînement en précision mixte (AMP)
from torch.cuda.amp import GradScaler, autocast

from src.dataset import CadCodeDataset
from src.model_enhanced import EncoderDecoderEnhanced

# --- Fonction utilitaire pour le batching ---
def collate_fn(batch):
    images, codes = zip(*batch)
    images = torch.stack(images, 0)
    max_len = max(len(c) for c in codes)
    padded_codes = torch.zeros(len(codes), max_len).long()
    for i, c in enumerate(codes):
        end = len(c)
        padded_codes[i, :end] = c[:end]
    return images, padded_codes

# --- Paramètres ---
TOKENIZER_PATH = 'tokenizer_vocab.json'
PARQUET_FILES = ['train_data_part1.parquet', 'train_data_part2.parquet']
DATA_FRACTION = 0.25
BATCH_SIZE = 32  # Taille de batch réduite pour éviter l'erreur "Out of Memory"
HIDDEN_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 3e-4

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    dataset = CadCodeDataset(
        parquet_files=PARQUET_FILES,
        tokenizer_path=TOKENIZER_PATH,
        fraction=DATA_FRACTION
    )
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=4,
        persistent_workers=True
    )
    
    model = EncoderDecoderEnhanced(
        embed_size=HIDDEN_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=dataset.vocab_size
    ).to(device)
    
    print("Compilation du modèle (peut prendre une minute)...")
    model = torch.compile(model)
    print("Compilation terminée.")

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- CORRECTION ICI ---
    # On revient à l'ancienne syntaxe qui fonctionne
    scaler = GradScaler()

    print("\n--- Démarrage de l'entraînement TURBO (AMP + Compile) ---")
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (images, codes) in enumerate(progress_bar):
            images, codes = images.to(device), codes.to(device)
            codes = codes.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # --- CORRECTION ICI ---
            # On revient à l'ancienne syntaxe qui fonctionne
            with autocast():
                targets = codes[:, 1:]
                decoder_inputs = codes[:, :-1]
                outputs = model(images, decoder_inputs)
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            progress_bar.set_postfix(loss=loss.item())
        
        print(f"---> Fin Epoch [{epoch+1}/{EPOCHS}]")
        
    # On sauvegarde le modèle en utilisant ._orig_mod pour un modèle compilé
    torch.save(model._orig_mod.state_dict(), 'enhanced_model_turbo.pth')
    print("\nModèle TURBO sauvegardé dans 'enhanced_model_turbo.pth'")

if __name__ == '__main__':
    main()