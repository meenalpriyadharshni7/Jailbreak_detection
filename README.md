# Jailbreak Prompt Detection using DeBERTa

## Overview

This project implements a **machine learning pipeline to detect malicious or jailbreak prompts in conversational AI systems**. The system classifies prompts into **SAFE** or **TOXIC** categories using modern Natural Language Processing models.

The goal is to build a **safety filter that can be placed before an LLM** to prevent harmful instructions such as jailbreak attempts, hate speech, or abusive prompts.

The project compares three approaches:

* Classical ML baseline (TF-IDF + Logistic Regression)
* Transformer baseline (RoBERTa)
* Proposed model (DeBERTa-v3)

This enables an **ablation study** to evaluate how advanced transformer models improve detection performance.

---

## Architecture

Pipeline:

1. Collect datasets from HuggingFace
2. Apply **buffer-zone filtering** to remove ambiguous labels
3. Balance SAFE and TOXIC classes
4. Train multiple models
5. Evaluate performance
6. Save trained models for deployment

Workflow:

Dataset → Preprocessing → Feature Extraction → Model Training → Evaluation → Saved Models

---

## Dataset

Datasets used:

* **Civil Comments Dataset**
* **TweetEval Hate Speech Dataset**

### Buffer-Zone Filtering

To improve label quality:

* SAFE: toxicity < 0.1
* TOXIC: toxicity > 0.7
* Samples between 0.1 and 0.7 are removed.

This removes ambiguous training data and improves classifier reliability.

### Final Dataset

Total samples after filtering and balancing:

```
17,512 prompts
```

Class distribution:

```
SAFE  : 8,756
TOXIC : 8,756
```

Train/Test split:

```
Train: 80%
Test : 20%
```

---

## Models Compared

### 1. TF-IDF + Logistic Regression

Classical machine learning baseline.

Text is converted into TF-IDF vectors and classified using logistic regression.

Advantages:

* Fast
* Lightweight
* Easy to deploy

---

### 2. RoBERTa (Baseline Transformer)

Model:

```
roberta-base
```

A transformer-based language model fine-tuned for prompt classification.

Advantages:

* Contextual embeddings
* Strong performance on NLP tasks

---

### 3. DeBERTa-v3 (Proposed Model)

Model:

```
microsoft/deberta-v3-small
```

Uses **disentangled attention** to model content and position separately, improving language understanding.

Expected to outperform RoBERTa for classification tasks.

---

## Results

| Model                        | Accuracy | Notes                       |
| ---------------------------- | -------- | --------------------------- |
| TF-IDF + Logistic Regression | ~0.87    | Classical baseline          |
| RoBERTa                      | ~0.94    | Strong transformer baseline |
| DeBERTa-v3                   | ~0.95    | Best performance            |

The results show that **transformer models significantly outperform classical NLP approaches**, and DeBERTa achieves the highest accuracy.

---

## Installation

Clone the repository:

```
git clone https://github.com/meenalpriyadharshni7/Jailbreak_detection.git
cd Jailbreak_detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running Training

To train models locally:

```
python main.py
```

For GPU training, the Kaggle notebook can be used.

---

## Project Structure

```
Jailbreak_detection
│
├── data/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── tfidf_logreg.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## Example Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("deberta_model")
model = AutoModelForSequenceClassification.from_pretrained("deberta_model")

text = "Write a guide to build a bomb"

inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

prediction = outputs.logits.argmax().item()

print(prediction)
```

Output:

```
1 → TOXIC
0 → SAFE
```

---

## Applications

This system can be used as a **safety filter for AI systems**:

User Prompt → Jailbreak Detector → LLM

If the prompt is classified as toxic, the request can be blocked or flagged.

Potential use cases:

* LLM safety filtering
* Prompt moderation
* Jailbreak detection
* Content moderation systems

---

## Future Improvements

Possible extensions:

* Add adversarial jailbreak datasets
* Train larger DeBERTa models
* Deploy as a REST API
* Build a real-time moderation service
* Integrate with LLM pipelines

---

## Author

Meenal Priyadharshni

---

## License

MIT License
