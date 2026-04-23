# Urdu Hate Speech Detection — NLP Assignment 3

**Author:** Hania Khan  
**Roll Number:** BS-23-IB-103973

---

## 📋 Overview

This project implements and compares multiple machine learning and deep learning models for **Urdu Hate Speech Detection**. The task is a text classification problem where the goal is to classify Urdu text into different categories: **hate**, **offensive**, and **normal** (or binary: hate vs. not-hate).

The project demonstrates a comprehensive NLP pipeline including data preprocessing, feature engineering, model training, and evaluation across 7 different models.

---

## 🎯 Objectives

1. Preprocess Urdu text with language-specific normalization
2. Implement and train traditional ML models:
   - Binary Naive Bayes (BernoulliNB / ComplementNB)
   - Multinomial Naive Bayes
   - Logistic Regression (Binary & Multi-class)
3. Implement and train deep learning models:
   - RNN (Recurrent Neural Network)
   - LSTM (Long Short-Term Memory)
   - GRU (Gated Recurrent Unit)
4. Compare model performance using standard metrics (Accuracy, Precision, Recall, F1-Score)

---

## 🛠️ Models Implemented

| Model | Type | Classification | Features |
|-------|------|---|---|
| **Binary Naive Bayes** | Machine Learning | Binary | ComplementNB + Binary TF-IDF |
| **Multinomial Naive Bayes** | Machine Learning | Multi-class (3) | TF-IDF (word + char n-grams) |
| **Logistic Regression (Binary)** | Machine Learning | Binary | TF-IDF (word + char n-grams) |
| **Logistic Regression (Multi-class)** | Machine Learning | Multi-class (3) | TF-IDF (word + char n-grams) |
| **RNN (Bidirectional SimpleRNN)** | Deep Learning | Multi-class (3) | Word Embeddings + Padding |
| **LSTM (Bidirectional)** | Deep Learning | Multi-class (3) | Word Embeddings + Padding |
| **GRU (Bidirectional)** | Deep Learning | Multi-class (3) | Word Embeddings + Padding |

---

## 📊 Dataset Characteristics

- **Language:** Urdu
- **Classes:** 3 (hate, offensive, normal)
- **Split Ratio:** 80% train, 10% validation, 10% test (stratified)
- **Class Balancing:** SMOTE applied to training data for imbalanced datasets
- **Preprocessing:** Urdu-specific normalization (diacritics removal, character mapping, etc.)

---

## 🔄 Pipeline Overview

### 1. **Installation & Dependencies**
```python
numpy==1.26.4, pandas, scikit-learn, imbalanced-learn, 
matplotlib, seaborn, tensorflow
```

### 2. **Data Loading & Exploration**
- Load Urdu dataset
- Analyze class distribution
- Explore sample texts

### 3. **Preprocessing**
- URL and HTML tag removal
- @mention and #hashtag removal
- Emoji removal
- Number removal
- Urdu-specific character normalization
- Extra whitespace removal
- Urdu script validation

### 4. **Label Encoding**
- **Binary labels:** hate + offensive → "hate", normal → "not_hate"
- **Multi-class labels:** 0 (hate), 1 (normal), 2 (offensive)

### 5. **Feature Engineering**
**For ML Models:**
- TF-IDF Vectorization with word n-grams (1-3)
- TF-IDF Vectorization with character n-grams (2-5)
- Combined word + character features

**For DL Models:**
- Tokenization (vocabulary size: 30,000)
- Sequence padding (max length: 128)
- Word Embeddings (dimension: 128)

### 6. **Model Training & Evaluation**
- Train each model on processed data
- Validate on validation set
- Test on held-out test set
- Generate confusion matrices
- Calculate metrics: Accuracy, Precision, Recall, F1-Score

---

## 📈 Evaluation Metrics

Each model is evaluated using:

- **Accuracy:** Overall correct predictions
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of Precision and Recall
- **Confusion Matrix:** Visual representation of predictions vs. ground truth

---

## 🚀 How to Run

### Prerequisites
```bash
python >= 3.8
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn tensorflow
```

### Execution
1. Open the Jupyter Notebook: `HaniaKhan_BS_23_IB_103973.ipynb`
2. Run cells sequentially:
   - Install dependencies
   - Load and preprocess data
   - Train and evaluate each model
   - Generate comparison plots

---

## 📁 Project Structure

```
NLPAssignment03/
├── HaniaKhan_BS_23_IB_103973.ipynb    # Main Jupyter Notebook
├── README.md                           # This file
└── (Optional outputs)
    └── Urdu_processed.csv             # Processed dataset
    └── model_comparison.png           # Comparison chart
```

---

## 🎓 Key Learnings

1. **Urdu Text Preprocessing:** Handling Urdu script normalization, diacritics, and character mappings
2. **Feature Engineering:** Combining word and character-level n-grams for better representation
3. **Class Imbalance:** Using SMOTE for handling imbalanced datasets
4. **Model Comparison:** Understanding trade-offs between ML and DL approaches
5. **Deep Learning:** Implementing bidirectional RNN, LSTM, and GRU architectures

---

## 📝 Notes

- **Random Seed:** 42 (for reproducibility)
- **Class Weights:** Balanced weights applied to handle class imbalance
- **Early Stopping:** Applied to deep learning models to prevent overfitting
- **Learning Rate Scheduling:** ReduceLROnPlateau callback for adaptive learning

---

## 📚 References

- TensorFlow/Keras Documentation
- Scikit-learn Library
- Imbalanced-learn (SMOTE)
- Urdu NLP Best Practices

---

## 📧 Contact

For questions or suggestions about this project, please reach out to the author.

---

**Last Updated:** April 2026
