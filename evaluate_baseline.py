# Fichier: evaluate_baseline.py
import torch
import json
import pandas as pd
from PIL import Image
from torchvision import transforms
from src.model_baseline import EncoderDecoderBaseline
# On importe les fonctions de métrique fournies dans le test
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best

# --- Paramètres ---
MODEL_PATH = 'baseline_model.pth'
TOKENIZER_PATH = 'tokenizer_vocab.json'
TEST_DATA_PATH = 'train_data_part1.parquet' # On peut utiliser un morceau des données pour l'éval
NUM_SAMPLES_TO_EVAL = 10 # Évaluer sur 10 exemples pour aller vite
EMBED_SIZE = 256
HIDDEN_SIZE = 512

def generate_code_from_image(model, image_tensor, tokenizer, device, max_length=500):
    """Génère du code à partir d'une seule image, token par token."""
    model.eval() # Passer en mode évaluation
    
    char_to_int = tokenizer['char_to_int']
    int_to_char = tokenizer['int_to_char']
    
    sos_token = char_to_int['<SOS>']
    eos_token = char_to_int['<EOS>']
    
    # Préparer l'image
    image_tensor = image_tensor.to(device).unsqueeze(0) # Ajouter une dimension de batch
    
    # Encoder l'image pour obtenir le features vector
    with torch.no_grad():
        features = model.encoder(image_tensor)
    
    # Commencer la génération avec le token <SOS>
    generated_tokens = [sos_token]
    
    for _ in range(max_length):
        # Préparer l'input pour le décodeur
        decoder_input = torch.LongTensor([generated_tokens[-1]]).to(device)
        
        with torch.no_grad():
            # Pour le premier token, l'état caché est le vecteur de l'image
            # Pour les suivants, le GRU gère son propre état caché interne
            if len(generated_tokens) == 1:
                outputs = model.decoder(features, decoder_input.unsqueeze(0))
            else:
                # Attention: cette baseline simple réutilise le feature vector à chaque pas
                # C'est une simplification, mais fonctionnelle pour une baseline.
                # Pour les pas suivants, on ne passe que le dernier token généré
                # (le GRU a une mémoire implicite de son état précédent)
                # Une meilleure approche passerait l'état caché, mais restons simple.
                
                # Créer un tenseur de la séquence actuelle pour le décodeur
                current_sequence = torch.LongTensor([generated_tokens]).to(device)
                outputs = model.decoder(features, current_sequence)

        # On ne s'intéresse qu'à la prédiction pour le dernier token
        last_prediction = outputs[:, -1, :]
        predicted_token_id = last_prediction.argmax(1).item()
        
        generated_tokens.append(predicted_token_id)
        
        # Arrêter si on prédit la fin de la séquence
        if predicted_token_id == eos_token:
            break
            
    # Convertir les IDs des tokens en caractères
    generated_code = "".join([int_to_char[str(token_id)] for token_id in generated_tokens[1:-1]]) # Exclure <SOS> et <EOS>
    return generated_code

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    # Charger le tokenizer
    with open(TOKENIZER_PATH, 'r') as f:
        tokenizer = json.load(f)
    vocab_size = len(tokenizer['char_to_int'])

    # Charger le modèle de baseline
    model = EncoderDecoderBaseline(EMBED_SIZE, HIDDEN_SIZE, vocab_size).to(device)
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

    print(f"\n--- Génération et évaluation sur {NUM_SAMPLES_TO_EVAL} échantillons ---")
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
    print("\n--- RÉSULTATS DES MÉTRIQUES ---")
    
    # Valid Syntax Rate
    vsr = evaluate_syntax_rate_simple(predicted_codes)
    print(f"Valid Syntax Rate (VSR): {vsr:.2f}")
    
    # Intersection over Union (IoU)
    # Note: cette fonction peut être lente
    ious = []
    for sample_id in predicted_codes:
        try:
            iou = get_iou_best(ground_truth_codes[sample_id], predicted_codes[sample_id])
            ious.append(iou)
        except Exception as e:
            # Si le code généré est trop mauvais pour créer un solide, l'IoU est 0
            print(f"Impossible de calculer l'IOU pour {sample_id}: {e}")
            ious.append(0.0)
            
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    print(f"Mean Best IoU: {mean_iou:.4f}")


if __name__ == '__main__':
    import io
    main()