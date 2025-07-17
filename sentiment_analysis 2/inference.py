import torch
from torch.nn.functional import softmax

def predict_sentiment(model, vocab, text, device, max_len=200):
    tokenizer = lambda x: x.lower().split()
    tokens = tokenizer(text)[:max_len]
    ids = vocab(tokens)
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    x = torch.tensor(ids).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = softmax(logits, dim=1).cpu().squeeze().tolist()
    return {"negative": probs[0], "positive": probs[1]}