# Sentiment Analysis with GRU and Transformer from Scratch

This project is a clean, modular implementation of sentiment analysis using two types of neural networks:
- **GRU-based Recurrent Neural Network**
- **Transformer-based Neural Network**

We use the **IMDB Movie Reviews Dataset** from TorchText.

---

## 📦 Project Structure

```
sentiment_analysis/
│
├── dataset.py              # Loads and tokenizes IMDB dataset
├── model_gru.py            # GRU-based sentiment model
├── model_transformer.py    # Transformer-based sentiment model
├── train.py                # Training loop
├── evaluate.py             # Evaluation logic
├── inference.py            # Inference function for new text
├── main.py                 # CLI training & inference pipeline
├── notebook.ipynb          # Explanation and demo (Jupyter)
└── README.md               # Project presentation
```

---

## ⚙️ Setup

```bash
pip install torch torchtext
python main.py
```

---

## ✅ Results

Sample output after training 3 epochs with GRU:

```
Epoch 1: Train Loss=0.5664, Test Loss=0.4982, Accuracy=0.7493
Epoch 2: Train Loss=0.3741, Test Loss=0.4687, Accuracy=0.7794
Epoch 3: Train Loss=0.2792, Test Loss=0.4913, Accuracy=0.7746
```

---

## 🔍 Observations

- GRU achieves good performance in fewer epochs.
- Transformer needs more regularization (dropout, longer training).
- Accuracy surpasses 75% with minimal tuning.
- Modular code allows easy switching between GRU and Transformer.

---

## 🧠 Inference

After training, the system prompts for user input:
```text
Enter a review (or 'exit'):
```
Output:
```json
{"negative": 0.12, "positive": 0.88}
```

---

## 📝 Future Improvements

- Add validation split and early stopping
- Integrate Dropout, LayerNorm for Transformer
- Add tokenizer serialization and inference endpoint (Flask/FastAPI)