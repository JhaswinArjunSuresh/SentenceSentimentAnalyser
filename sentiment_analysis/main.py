import torch
from dataset import build_dataset
from model_gru import GRUSentiment
# from model_transformer import TransformerSentiment
from train import train
from evaluate import evaluate
from inference import predict_sentiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_dim = 128
hidden_dim = 256
output_dim = 2
num_heads = 4
num_layers = 2
epochs = 3

train_loader, test_loader, vocab = build_dataset()
vocab_size = len(vocab)

model = GRUSentiment(vocab_size, embed_dim, hidden_dim, output_dim).to(device)
# model = TransformerSentiment(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, output_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, device)
    acc, test_loss = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Accuracy={acc:.4f}")

while True:
    text = input("Enter a review (or 'exit'): ")
    if text.lower() == 'exit':
        break
    result = predict_sentiment(model, vocab, text, device)
    print("Sentiment Score:", result)