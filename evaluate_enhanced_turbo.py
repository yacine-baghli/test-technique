import torch
import torch.nn as nn
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
MODEL_PATH = 'enhanced_model_turbo.pth'
TOKENIZER_PATH = 'tokenizer_vocab.json'
TEST_DATA_PATH = 'train_data_part1.parquet'
NUM_SAMPLES_TO_EVAL = 10 
HIDDEN_SIZE = 512
BEAM_WIDTH = 3 # Largeur du faisceau pour le Beam Search

def generate_code_beam_search(model, image_tensor, tokenizer, device, max_length=500, beam_width=3):
    """Génère du code à partir d'une seule image en utilisant la technique du Beam Search."""
    model.eval()
    
    char_to_int = tokenizer['char_to_int']
    int_to_char = {int(k): v for k, v in tokenizer['int_to_char'].items()}
    
    sos_token_id = char_to_int['<SOS>']
    eos_token_id = char_to_int['<EOS>']
    
    image_tensor = image_tensor.to(device).unsqueeze(0)
    
    with torch.no_grad():
        features = model.encoder(image_tensor)

    # Initialiser les "faisceaux" (beams)
    # Chaque faisceau est une tuple (séquence, probabilité_logarithmique)
    beams = [([sos_token_id], 0.0)]
    
    for _ in range(max_length):
        all_candidates = []
        for seq, score in beams:
            if seq[-1] == eos_token_id:
                all_candidates.append((seq, score))
                continue

            decoder_input = torch.LongTensor([seq]).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor, decoder_input)
            
            last_prediction = outputs[:, -1, :]
            log_probs = torch.nn.functional.log_softmax(last_prediction, dim=-1)
            
            top_log_probs, top_indices = log_probs.topk(beam_width)
            
            for i in range(beam_width):
                new_seq = seq + [top_indices[0, i].item()]
                new_score = score + top_log_probs[0, i].item()
                all_candidates.append((new_seq, new_score))

        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = ordered[:beam_width]
        
        if beams[0][0][-1] == eos_token_id:
            break
            
    best_seq = beams[0][0]
    generated_code = "".join([int_to_char.get(token_id, '?') for token_id in best_seq[1:-1]])
    return generated_code

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    with open(TOKENIZER_PATH, 'r') as f:
        tokenizer = json.load(f)
    vocab_size = len(tokenizer['char_to_int'])

    model = EncoderDecoderEnhanced(HIDDEN_SIZE, HIDDEN_SIZE, vocab_size).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"ERREUR : Fichier modèle non trouvé : '{MODEL_PATH}'")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Modèle '{MODEL_PATH}' chargé.")

    df_test = pd.read_parquet(TEST_DATA_PATH).sample(NUM_SAMPLES_TO_EVAL, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    ground_truth_codes = {}
    predicted_codes = {}

    print(f"\n--- Génération avec Beam Search (width={BEAM_WIDTH}) sur {NUM_SAMPLES_TO_EVAL} échantillons ---")
    for index, row in df_test.iterrows():
        image_data = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
        image_tensor = transform(image_data)
        
        true_code = row['cadquery']
        
        predicted_code = generate_code_beam_search(model, image_tensor, tokenizer, device, beam_width=BEAM_WIDTH)
        
        sample_id = f"sample_{index}"
        ground_truth_codes[sample_id] = true_code
        predicted_codes[sample_id] = predicted_code
        
        print(f"\n--- {sample_id} ---")
        print(f"VRAI CODE:\n{true_code}")
        print(f"CODE PRÉDIT:\n{predicted_code}")

    print("\n--- RÉSULTATS DES MÉTRIQUES (BEAM SEARCH) ---")
    vsr = evaluate_syntax_rate_simple(predicted_codes)
    print(f"Valid Syntax Rate (VSR): {vsr:.2f}")
    
    print("Calcul de l'IoU moyen...")
    ious = []
    for sample_id in tqdm(predicted_codes, desc="Calcul IoU"):
        if predicted_codes[sample_id] and evaluate_syntax_rate_simple({sample_id: predicted_codes[sample_id]}) == 1.0:
            try:
                iou = get_iou_best(ground_truth_codes[sample_id], predicted_codes[sample_id])
                ious.append(iou)
            except Exception:
                ious.append(0.0)
        else:
            ious.append(0.0)
            
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    print(f"Mean Best IoU: {mean_iou:.4f}")

if __name__ == '__main__':
    main()