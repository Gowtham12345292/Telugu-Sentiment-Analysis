## Telugu Sentiment Analysis — Fine-Tuned mBERT

Fine-tuned `bert-base-multilingual-cased` (mBERT) on Telugu sentiment data for 3-class classification (Positive, Negative, Neutral). Deployed on HuggingFace with real-time Streamlit interface.

Live Model: [HuggingFace — Gowthamvemula/Teugu_Sentimental_fine-tuning](https://huggingface.co/Gowthamvemula/Teugu_Sentimental_fine-tuning)

---

## Problem Statement

Sentiment analysis for regional Indian languages like Telugu is underserved. Most NLP models focus on English. This project fine-tunes a multilingual Transformer model to accurately classify Telugu text into Positive, Negative, or Neutral sentiment. This enables applications in social media monitoring, customer feedback analysis, and regional content understanding.

---

## Architecture

```
Telugu Text Input
       |
AutoTokenizer (bert-base-multilingual-cased)
  - max_length: 512
  - truncation + padding
       |
Fine-Tuned mBERT (3 labels)
  - Learning Rate: 2e-5
  - Epochs: 3 (with Early Stopping, patience=2)
  - Batch Size: 8
  - FP16 Mixed Precision
       |
Sentiment Prediction
  [Neutral | Positive | Negative]
       |
Streamlit App / HuggingFace API
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | google-bert/bert-base-multilingual-cased |
| Dataset | mounikaiiith/Telugu_Sentiment (HuggingFace Datasets) |
| Framework | HuggingFace Transformers, PyTorch |
| Training | HuggingFace Trainer API |
| Deployment | HuggingFace Model Hub |
| Frontend | Streamlit |
| Language | Python |

---

## Key Results

| Metric | Value |
|--------|-------|
| F1-Score | 87% |
| Accuracy | 89% |
| Classes | Neutral, Positive, Negative |
| Training Samples | 50,000+ |
| Inference Latency | sub-300ms |

---

## How to Run

### Option 1: Use the Deployed Model
```python
from transformers import pipeline

pipe = pipeline("text-classification", model="Gowthamvemula/Teugu_Sentimental_fine-tuning")

result = pipe("ఈ సినిమా చాలా బాగుంది")  # "This movie is very good"
print(result)
# Output: [{'label': 'LABEL_1', 'score': 0.95}]
# LABEL_0 = Neutral, LABEL_1 = Positive, LABEL_2 = Negative
```

### Option 2: Run Locally
```bash
# Clone the repo
git clone https://github.com/Gowtham12345292/Telugu-Sentiment-Analysis.git
cd Telugu-Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### Option 3: Load Model Directly
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Gowthamvemula/Teugu_Sentimental_fine-tuning")
model = AutoModelForSequenceClassification.from_pretrained("Gowthamvemula/Teugu_Sentimental_fine-tuning")
```

---

## Project Structure
```
├── README.md
├── Telugu_Sentiment_Analysis.ipynb    # Training notebook
├── app.py                             # Streamlit frontend
├── requirements.txt                   # Dependencies
└── .gitignore
```

---

## Training Configuration

```python
TrainingArguments(
    learning_rate = 2e-5,
    num_train_epochs = 3,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    fp16 = True,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss"
)

EarlyStoppingCallback(early_stopping_patience=2)
```

Dataset: mounikaiiith/Telugu_Sentiment from HuggingFace. 3 classes (neutral, pos, neg). Preprocessed with null handling and label encoding using ClassLabel. Tokenized with max_length=512, padding, and truncation.

---

## Example Predictions

| Telugu Text | English Translation | Prediction |
|-------------|-------------------|------------|
| ఈ సినిమా చాలా బాగుంది | This movie is very good | Positive |
| ఈ ఆహారం చాలా చెడుగా ఉంది | This food is very bad | Negative |
| నాకు ఈ రోజు చాలా సంతోషంగా ఉంది | I am very happy today | Positive |
| నేను ఈ వార్తలకు చాలా బాధపడ్డాను | I felt very sad for this news | Negative |

---

## Design Decisions

1. mBERT over base BERT: Telugu is not in BERT's vocabulary. mBERT is pretrained on 104 languages including Telugu.
2. Learning Rate 2e-5: Standard for fine-tuning Transformers. Prevents catastrophic forgetting of pretrained weights.
3. FP16 Mixed Precision: Halves memory usage and speeds up training without accuracy loss.
4. Early Stopping (patience=2): Prevents overfitting. Stops training if validation loss doesn't improve for 2 consecutive epochs.
5. Max Length 512: Telugu sentences can be long. Using full BERT context window for better understanding.

---

## What I Learned

- Fine-tuning pretrained Transformers for low-resource languages
- HuggingFace Trainer API and training pipeline
- Handling multilingual tokenization and null values in datasets
- Deploying models to HuggingFace Hub for real-time inference
- Building Streamlit interfaces for ML model demos

---

## Contact

Vemula Gowtham — [LinkedIn](https://linkedin.com/in/vemula-gowtham-624206286) | vemulagowtham7@gmail.com
