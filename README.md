# Misogyny Detection in Memes

A machine learning project that classifies meme text content as misogynous or non-misogynous using various text representation techniques.

## Overview

This notebook explores different approaches to detecting misogyny in meme text data. We compare traditional bag-of-words methods (TF-IDF) with word embedding techniques (FastText) to understand which features best capture misogynistic language patterns.

## Project Structure

```
.
├── data/              # Raw and processed data files
│   ├── training.csv  # Training dataset (TSV format)
│   └── README.md     # Data documentation
├── notebooks/         # Jupyter notebooks with analysis
│   └── Lab1.ipynb    # Main analysis notebook
├── .gitignore        # Git ignore rules
├── LICENSE           # MIT License
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

## Dataset

- **Total samples**: 7,500 meme texts
- **Training set**: 6,000 samples (80%)
- **Test set**: 1,500 samples (20%)
- **Labels**: Binary classification (0 = non-misogynous, 1 = misogynous)
- **Split strategy**: Stratified random sampling to maintain class balance

## Methodology

### Text Representation Approaches

**1. Manual TF-IDF Implementation**
- Built from scratch to understand the underlying mechanics
- Vocabulary size: 21,079 unique terms
- F1-Score: 0.7785

**2. Scikit-learn TF-IDF**
- Optimized implementation with dimensionality reduction
- Limited to 5,000 most frequent features
- F1-Score: 0.7840 (best overall)

**3. FastText Embeddings**
- Dense 100-dimensional word vectors
- Captures semantic relationships between words
- F1-Score: 0.3357 (significantly underperformed)

### Experiments Conducted

**N-gram Analysis**
- Unigrams: F1 = 0.7840
- Bigrams: F1 = 0.7873
- Trigrams: F1 = 0.7873

Bigrams slightly improved performance by capturing short phrases.

**Preprocessing Impact**
- Raw text: F1 = 0.7840
- Lowercase: F1 = 0.7840
- Stopword removal: F1 = 0.7846 (best)
- Lemmatization: F1 = 0.7813

Minimal preprocessing worked best, suggesting that stopwords and punctuation carry useful information in this context.

**Hyperparameter Optimization**
- Regularization parameter C = 1.0 proved optimal
- Used 5-fold cross-validation for robust evaluation

## Results

### Best Model Performance

Using TF-IDF with stopword removal and Logistic Regression (C=1):

**Confusion Matrix:**
```
                 Predicted: Non-Mis    Predicted: Mis
Actual: Non-Mis          599                143
Actual: Mis              181                577
```

**Metrics:**
- Precision: 80.1%
- Recall: 76.1%
- F1-Score: 0.7846
- Overall accuracy: 78.4%

### Key Findings

**Why TF-IDF outperformed FastText?**

TF-IDF worked better because:
1. Misogyny detection relies heavily on specific keyword presence
2. FastText's semantic averaging dilutes discriminative signals
3. The training corpus may be too small for effective embedding learning
4. Context averaging loses information about specific toxic terms

**Most Discriminative Features:**

Top misogynous indicators:
- employees (+5.12)
- wouldn (+4.59)
- fits (+4.39)
- attracted (+4.01)

Top non-misogynous indicators:
- hate (-4.11)
- buys (-2.93)
- clientis (-2.81)

**Error Analysis:**

False Positives (143 cases):
- Empathetic contexts misunderstood (e.g., humanizing sex workers)
- Feminist content mentioning sensitive topics
- Sarcasm used to criticize misogyny

False Negatives (181 cases):
- Subtle, implicit stereotypes without offensive keywords
- Ironic minimization of sexism complaints
- Sexual insinuations without explicit language
- Cultural assumptions about gender roles

## Limitations

1. **Context blindness**: Bag-of-words cannot distinguish between mentioning a problem and promoting it
2. **Sarcasm detection**: Model struggles with multiple layers of irony
3. **Implicit bias**: Fails to catch subtle stereotypes without obvious keywords
4. **Cultural nuance**: Cannot understand social norms and power dynamics

## Future Improvements

**Short term:**
- Collect more annotated examples of subtle misogyny
- Experiment with character n-grams for robustness to typos
- Implement class weighting to penalize false negatives more heavily

**Long term:**
- Use transformer models (BERT, RoBERTa) for contextual understanding
- Incorporate multimodal analysis combining image and text
- Apply transfer learning from large hate speech corpora
- Build ensemble methods combining multiple approaches

## Technical Stack

- Python 3.12
- pandas, numpy for data manipulation
- scikit-learn for classical ML
- FastText for word embeddings
- NLTK for text preprocessing
- matplotlib for visualization

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Krypto02/ComputerVision.git
cd ComputerVision
```

2. Install dependencies (choose one method):
:
```bash
pip install -r requirements.txt

## Usage

1. Place your data files in the `data/` directory
2. Open the notebook:
```bash
jupyter notebook notebooks/Lab1.ipynb
```

The notebook is self-contained and executes sequentially. All cells include detailed explanations of the methodology, implementation details, and interpretation of results.

Simply run all cells in order to reproduce the analysis.

## Conclusion

For this dataset, simpler is better. TF-IDF with minimal preprocessing and logistic regression achieved nearly 78.5% F1-score, substantially outperforming more sophisticated embedding methods. The model performs well on obvious cases but struggles with subtle misogyny that requires deep contextual understanding, suggesting that future work should focus on transformer-based architectures capable of capturing nuanced language patterns.
