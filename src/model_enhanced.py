# Fichier: src/model_enhanced.py
import torch
import torch.nn as nn
import torchvision.models as models
import math

# L'Encoder ne change pas, on peut le réutiliser
class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        resnet = models.resnet34(weights='ResNet34_Weights.DEFAULT')
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.fc(features))
        return features

# --- NOUVEAU DÉCODEUR TRANSFORMER ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=4, nhead=8):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=hidden_size, batch_first=True),
            num_layers=num_layers
        )
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        # features: [batch, hidden_size], captions: [batch, seq_len]
        # Note: pour un vrai Transformer, 'features' devrait être la sortie complète de l'encoder.
        # Ici, on le garde simple en le traitant comme une "mémoire"
        memory = features.unsqueeze(1) # [batch, 1, hidden_size]
        
        embeddings = self.embedding(captions)
        # On doit avoir la même dim que memory pour le decoder
        if embeddings.shape[2] != memory.shape[2]:
            # Simple projection si les tailles ne correspondent pas
            proj = nn.Linear(embeddings.shape[2], memory.shape[2]).to(embeddings.device)
            embeddings = proj(embeddings)
            
        # Créer le masque pour le "teacher forcing"
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions.size(1)).to(captions.device)
        
        output = self.transformer_decoder(embeddings, memory, tgt_mask=tgt_mask)
        return self.linear(output)

class EncoderDecoderEnhanced(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(EncoderDecoderEnhanced, self).__init__()
        self.encoder = Encoder(hidden_size)
        # On ajuste embed_size pour qu'il corresponde à hidden_size pour simplifier
        self.decoder = TransformerDecoder(hidden_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs