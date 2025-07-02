# Fichier: src/model_baseline.py (corrigé)
import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    # La seule modification est ici : l'encodeur doit connaître la 'hidden_size' du décodeur
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        resnet = models.resnet34(weights='ResNet34_Weights.DEFAULT')
        # Geler les poids du ResNet
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # On s'assure que la sortie de l'encodeur correspond à la 'hidden_size' du GRU
        self.fc = nn.Linear(resnet.fc.in_features, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.fc(features))
        return features

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # On utilise .clone() pour éviter les problèmes de modification in-place avec l'embedding
        embeddings = self.embedding(captions).clone() 
        # On utilise le vecteur de l'image comme état caché initial du GRU
        hiddens, _ = self.gru(embeddings, features.unsqueeze(0))
        outputs = self.linear(hiddens)
        return outputs

class EncoderDecoderBaseline(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(EncoderDecoderBaseline, self).__init__()
        # On passe la bonne taille à l'Encoder
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs