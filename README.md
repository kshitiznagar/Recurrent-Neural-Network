# Recurrent Neural Network (RNN) & CNN Experiments in PyTorch

Two small deep-learning projects built with **PyTorch**, written as Jupyter notebooks:

1. **Sentiment Analysis with an RNN**: classifies IMDB movie reviews as positive or negative.
2. **Digit Recognition with a CNN**: classifies handwritten digits from the MNIST dataset.

Both notebooks pick up Apple Silicon GPU acceleration (`mps`) when it is available and fall back to CPU otherwise.

---

## Repository Structure

```
Recurrent-Neural-Network/
├── Sentiment_Analysis_RNN.ipynb   # IMDB sentiment classification using an RNN
├── Digit_Recognition.ipynb        # MNIST digit classification using a CNN
├── IMDB Dataset.csv               # 50,000 labelled movie reviews
├── data/MNIST/raw/                # MNIST files (downloaded automatically by torchvision)
└── README.md
```

---

## 1. Sentiment Analysis with an RNN

**Notebook:** `Sentiment_Analysis_RNN.ipynb`
**Dataset:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (`review`, `sentiment` columns)

### Pipeline

| Step | Details |
|------|---------|
| Cleaning | Drop duplicates, lowercase, remove URLs, punctuation and HTML tags |
| NLP preprocessing | Tokenisation and stop-word handling with **NLTK**, then **Porter stemming** |
| Label encoding | `LabelEncoder` (negative → 0, positive → 1) |
| Vectorisation | `TfidfVectorizer` with `max_features=5000` |
| Split | 80% train / 20% test (`random_state=42`) |
| Data loading | `TensorDataset` + `DataLoader` (batch size 64) |

### Model

```
Input (5000 TF-IDF features)
   └── nn.RNN (hidden_size=128, num_layers=1, batch_first=True)
        └── Linear(128 → 1)
             └── Sigmoid → probability of positive sentiment
```

- **Loss:** Binary Cross-Entropy (`BCELoss`)
- **Optimizer:** Adam
- **Epochs:** 10

### Result

| Metric | Value |
|--------|-------|
| Test accuracy | **~87.3%** |

---

## 2. Digit Recognition with a CNN

**Notebook:** `Digit_Recognition.ipynb`
**Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/) (loaded through `torchvision.datasets.MNIST`)

### Pipeline

- Resize to 28×28, convert to tensor, normalise with mean 0.5 and std 0.5
- `DataLoader` with batch size 64

### Model

```
Conv2d(1 → 28, 3x3) → ReLU → MaxPool(2x2)
Conv2d(28 → 56, 3x3) → ReLU → MaxPool(2x2)
Flatten (7·7·56)
Linear(3136 → 112) → ReLU
Linear(112 → 10)
```

- **Loss:** Cross-Entropy
- **Optimizer:** Adam
- **Epochs:** 10

---

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/kshitiznagar/Recurrent-Neural-Network.git
cd Recurrent-Neural-Network

pip install torch torchvision pandas scikit-learn nltk jupyter
```

### Run

```bash
jupyter notebook
```

Then open either notebook and run the cells from top to bottom.

- The sentiment notebook downloads the required NLTK data (`punkt`, `punkt_tab`, `stopwords`) on first run.
- The MNIST dataset downloads automatically into `./data`.

---

## Tech Stack

- **Deep learning:** PyTorch, torchvision
- **NLP:** NLTK (tokenisation, stop words, Porter stemmer)
- **ML utilities:** scikit-learn (TF-IDF, label encoding, train/test split)
- **Data handling:** pandas
- **Environment:** Jupyter Notebook

---

## Known Limitations & Future Work

- **Sequence length of 1 in the RNN.** TF-IDF vectors are fed to the RNN as a single time step, so the recurrent layer does not model word order. Using an embedding layer with tokenised sequences (and an LSTM/GRU) would let the network make use of sequence information.
- **Stop-word removal has no effect.** In `remove_stop_words`, the result of `text.replace(...)` is not assigned back, so the text is returned unchanged.
- **Preprocessing order.** Punctuation is stripped before HTML tags are removed, which leaves `br` tokens from `<br />` in the reviews.
- **CNN training loss barely moves** (about 2.33 → 2.30, close to ln 10), which suggests the model is not learning effectively yet. A learning-rate check and a test-set evaluation cell would help here.
- Add a model-saving step, an inference example, and plots of loss and accuracy.

---
