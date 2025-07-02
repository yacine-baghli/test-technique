import torch
import json
import pandas as pd
from PIL import Image
from torchvision import transforms
import io
import os
from tqdm import tqdm

# On importe le modèle amélioré
from src.model_enhanced import EncoderDecoderEnhanced
# On importe les fonctions de métrique fournies dans le test
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best

# --- Paramètres ---
# On pointe vers le nouveau modèle
MODEL_PATH = 'enhanced_model.pth'
TOKENIZER_PATH = 'tokenizer_vocab.json'
TEST_DATA_PATH = 'train_data_part1.parquet'
NUM_SAMPLES_TO_EVAL = 10 # Évaluer sur 10 exemples pour aller vite
# Les hyperparamètres doivent correspondre au modèle entraîné
HIDDEN_SIZE = 512

def generate_code_from_image(model, image_tensor, tokenizer, device, max_length=500):
    """Génère du code à partir d'une seule image avec le modèle Transformer."""
    model.eval()
    
    char_to_int = tokenizer['char_to_int']
    int_to_char = {int(k): v for k, v in tokenizer['int_to_char'].items()} # Assurer que les clés sont des entiers
    
    sos_token_id = char_to_int['<SOS>']
    eos_token_id = char_to_int['<EOS>']
    
    # Préparer l'image
    image_tensor = image_tensor.to(device).unsqueeze(0)
    
    # Commencer la génération avec le token <SOS>
    generated_tokens = [sos_token_id]
    
    # Encoder l'image une seule fois
    with torch.no_grad():
        features = model.encoder(image_tensor)

    for _ in range(max_length):
        # Préparer la séquence actuelle pour le décodeur
        decoder_input = torch.LongTensor([generated_tokens]).to(device)
        
        with torch.no_grad():
            # Le modèle Transformer prend l'image et la séquence actuelle
            outputs = model(image_tensor, decoder_input)
        
        # On ne s'intéresse qu'à la prédiction pour le dernier token de la séquence
        last_prediction = outputs[:, -1, :]
        predicted_token_id = last_prediction.argmax(1).item()
        
        generated_tokens.append(predicted_token_id)
        
        # Arrêter si on prédit la fin de la séquence
        if predicted_token_id == eos_token_id:
            break
            
    # Convertir les IDs des tokens en caractères
    generated_code = "".join([int_to_char[token_id] for token_id in generated_tokens[1:-1]])
    return generated_code

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    # Charger le tokenizer
    with open(TOKENIZER_PATH, 'r') as f:
        tokenizer = json.load(f)
    vocab_size = len(tokenizer['char_to_int'])

    # Charger le modèle amélioré
    # On utilise HIDDEN_SIZE pour embed_size comme dans notre entraînement
    model = EncoderDecoderEnhanced(HIDDEN_SIZE, HIDDEN_SIZE, vocab_size).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"ERREUR : Fichier modèle non trouvé : '{MODEL_PATH}'")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Modèle '{MODEL_PATH}' chargé.")

    # Charger quelques données de test
    df_test = pd.read_parquet(TEST_DATA_PATH).sample(NUM_SAMPLES_TO_EVAL, random_state=42)

    # Préparer les transformations d'image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    ground_truth_codes = {}
    predicted_codes = {}

    print(f"\n--- Génération et évaluation du modèle amélioré sur {NUM_SAMPLES_TO_EVAL} échantillons ---")
    for index, row in df_test.iterrows():
        image_data = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
        image_tensor = transform(image_data)
        
        true_code = row['cadquery']
        
        # Générer le code à partir de l'image
        predicted_code = generate_code_from_image(model, image_tensor, tokenizer, device)
        
        sample_id = f"sample_{index}"
        ground_truth_codes[sample_id] = true_code
        predicted_codes[sample_id] = predicted_code
        
        print(f"\n--- {sample_id} ---")
        print(f"VRAI CODE:\n{true_code}")
        print(f"CODE PRÉDIT:\n{predicted_code}")

    # Évaluation
    print("\n--- RÉSULTATS DES MÉTRIQUES (MODÈLE AMÉLIORÉ) ---")
    
    # Valid Syntax Rate
    vsr = evaluate_syntax_rate_simple(predicted_codes)
    print(f"Valid Syntax Rate (VSR): {vsr:.2f}")
    
    # Intersection over Union (IoU)
    print("Calcul de l'IoU moyen (cela peut prendre un moment)...")
    ious = []
    for sample_id in tqdm(predicted_codes, desc="Calcul IoU"):
        # On ne calcule que si la syntaxe est valide pour gagner du temps
        if predicted_codes[sample_id] and evaluate_syntax_rate_simple({sample_id: predicted_codes[sample_id]}) == 1.0:
            try:
                iou = get_iou_best(ground_truth_codes[sample_id], predicted_codes[sample_id])
                ious.append(iou)
            except Exception as e:
                print(f"Impossible de calculer l'IOU pour {sample_id}: {e}")
                ious.append(0.0)
        else:
            ious.append(0.0)
            
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    print(f"Mean Best IoU: {mean_iou:.4f}")

if __name__ == '__main__':
    main()