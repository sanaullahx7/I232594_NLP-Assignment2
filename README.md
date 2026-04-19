# 🧠 Neural NLP Pipeline for BBC Urdu Corpus

**Course:** CS-4063 Natural Language Processing
**Assignment:** 2
**Student ID:** i23-2594
**Section:** DS-6A

---

## 📌 Overview

This project implements a complete Neural Natural Language Processing (NLP) pipeline for the BBC Urdu corpus. The pipeline is built entirely from scratch using PyTorch, without relying on pretrained models or high-level libraries such as Gensim or HuggingFace.

The system includes:

* Word Embeddings (TF-IDF, PPMI, Word2Vec)
* Sequence Labeling (POS Tagging and NER using BiLSTM-CRF)
* Topic Classification (Transformer Encoder)

---

## 📂 Project Structure

```
.
├── embeddings/
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy
│   └── word2idx.json
│
├── models/
│   ├── bilstm_pos.pt
│   ├── bilstm_ner.pt
│   └── transformer_cls.pt
│
├── data/
│   ├── pos_train.conll
│   ├── pos_test.conll
│   ├── ner_train.conll
│   └── ner_test.conll
│
└── scripts / notebooks
```

---

## 🧩 Dataset

* 245 BBC Urdu news articles
* Categories:

  * Sports
  * Politics
  * Economy
  * International
  * Health & Society

Preprocessing steps:

* Tokenization
* Cleaning
* Vocabulary size: 10,000 tokens

---

## 🔹 Part 1: Word Embeddings

### TF-IDF

* Sparse term-document matrix
* Strong discriminative power

### PPMI

* Window size: 5
* Matrix size: 10,000 × 10,000
* Captures co-occurrence relationships

### Skip-gram Word2Vec

Configuration:

* Embedding dimension: 100
* Window size: 5
* Negative samples: 10
* Batch size: 512
* Optimizer: Adam

Results:

* Final loss ≈ 0.45
* Captures semantic relationships
* Analogy accuracy: 60%



## 📊 Embedding Comparison

Condition C1 (PPMI): MRR = 0.58
Condition C2 (Skip-gram raw): MRR = 0.65
Condition C3 (Skip-gram cleaned): MRR = 0.72 (Best)
Condition C4 (Skip-gram d=200): MRR = 0.73

Conclusion: Preprocessing significantly improves embedding quality.

---

## 🔹 Part 2: Sequence Labeling

### Dataset

* 500 annotated sentences
* Split: 70% train / 15% validation / 15% test

### POS Tagging

* 12 tags
* Accuracy: 87.3%
* Macro-F1: 0.84

### Named Entity Recognition (NER)

PER: 0.78
LOC: 0.81
ORG: 0.74
MISC: 0.68
Overall F1: 0.76

Model:

* 2-layer BiLSTM
* Dropout: 0.5
* CRF layer improves performance by 4.2%



## 🔬 Ablation Study

Unidirectional LSTM → -6.8% F1
No dropout → -3.2% F1
Random embeddings → -8.5% F1
No CRF → -4.2% F1



## 🔹 Part 3: Transformer Classification

Architecture:

* 4 encoder layers
* 4 attention heads
* d_model = 128
* d_ff = 512
* Sequence length = 256

Training:

* Optimizer: AdamW
* Learning rate: 5e-4
* Epochs: 20

Results:

* Accuracy: 91.2%
* Macro-F1: 0.89


## ⚖️ BiLSTM vs Transformer

Accuracy: Transformer higher (91.2% vs 87.8%)
Speed: BiLSTM faster
Context: Transformer captures long-range dependencies
Interpretability: Transformer provides attention insights



## 🔍 Key Insights

* Preprocessing improves embeddings
* Word2Vec enhances downstream tasks
* CRF improves sequence labeling
* Transformer achieves best classification performance
* BiLSTM is more efficient for small datasets



## 🚀 How to Run

Install dependencies:
pip install torch numpy sklearn matplotlib

Run training:
python train_embeddings.py
python train_pos.py
python train_ner.py
python train_transformer.py


## 📈 Outputs

* Embeddings stored in /embeddings
* Models stored in /models
* Data stored in /data




## 🔗 Repository

[https://github.com/sanaullahx7/I232594_NLP-Assignment2](https://github.com/sanaullahx7/I232594_NLP-Assignment2)



## 👨‍💻 Author

Sanaullah
Student ID: i23-2594
FAST NUCES


