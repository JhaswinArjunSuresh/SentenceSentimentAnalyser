import torch.nn as nn
import torch

class TransformerSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x) * torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        embedded = embedded.permute(1, 0, 2)  # Transformer needs (seq, batch, dim)
        encoded = self.transformer_encoder(embedded)
        pooled = encoded.mean(dim=0)
        return self.fc(pooled)