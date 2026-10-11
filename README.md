# Classifying Academic Subjects Based On QA texts

A comprehensive Natural Language Processing (NLP) project that benchmarks classical Machine Learning and Neural Network architectures for multi-class text classification, using multiple word representation techniques.

---

## 📌 Overview

This project addresses **answer-type classification** in Question-Answering (QA) systems — determining whether a query seeks a numeric fact, a location, a person, an entity, or a descriptive explanation. Such classification allows downstream QA components to narrow their search space and tailor retrieval strategies.

The project systematically compares:

- **4 Word Representation Techniques:** Bag of Words (BoW), TF-IDF, GloVe, Skip-gram Word2Vec
- **3 Classical ML Models:** Logistic Regression, Multinomial Naive Bayes, Random Forest
- **7 Neural Network Models:** DNN, SimpleRNN, GRU, LSTM, Bi-SimpleRNN, Bi-GRU, Bi-LSTM

A total of **22+ separate training runs** were conducted, with manual hyperparameter tuning guided by validation performance.

---

## 📂 Dataset

- **Source:** Provided QA-type classification corpus (assigned dataset)
- **Size:** 280,003 samples
- **Split:** Pre-split into training (80%) and testing (20%) sets as separate CSV files
- **Classes:** 10 balanced categories
  - Entertainment & Music
  - Health
  - Sports
  - Science & Mathematics
  - Computers & Internet
  - Education & Reference
  - Business & Finance
  - Family & Relationships
  - Society & Culture
  - Politics & Government

> ⚠️ **Note:** The dataset is well-balanced, with each class containing approximately 10% of the total samples.

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights uncovered during EDA:

- **Class Distribution:** Perfectly balanced across all 10 classes (verified via bar chart and pie chart).
- **Word Cloud:** Most frequent terms include *"Question," "Title," "Content," "Best," "Answer."*
- **Text Length Distribution:** Most documents contain between **50–250 words**.
- **Correlation Analysis:** Heatmap shows a **0.99 correlation** between `text_length` and `word_count` — a strong linear relationship.

---

## 🧹 Data Preprocessing

The following preprocessing pipeline was applied (implemented as modular, reusable functions):

1. Lowercasing
2. Punctuation stripping
3. Stop-word removal
4. Tokenization
5. Lemmatization

---

## 🧪 Methodology

### Word Representations

| Technique | Description |
|-----------|-------------|
| **BoW** | Unigram counts with min-df and max-features limits |
| **TF-IDF** | Term frequency–inverse document frequency weighting |
| **GloVe** | Pre-trained 200d embeddings (frozen/fine-tuned) |
| **Skip-gram** | Custom Word2Vec embeddings trained on the corpus (trigram implemented) |

### Models & Experiment Matrix

| Representation | ML Models | NN Models |
|----------------|-----------|-----------|
| BoW | LR, NB, RF, DNN | — |
| TF-IDF | LR, NB, RF, DNN | — |
| GloVe | — | DNN, SimpleRNN, GRU, LSTM, Bi-SimpleRNN, Bi-GRU, Bi-LSTM |
| Skip-gram | — | DNN, SimpleRNN, GRU, LSTM, Bi-SimpleRNN, Bi-GRU, Bi-LSTM |

### Training Protocol

- **Optimizer:** Adam (initial LR = 1e-3)
- **Gradient Clipping:** 1.0 (to stabilize RNN training)
- **Batch Size:** 64–128
- **Epochs:** Up to 20–30 with early stopping (patience 3–5)
- **Regularization:** Dropout on embeddings/recurrent outputs; L2 on dense weights
- **Validation Split:** 10–20% of training data for hyperparameter selection

### Evaluation Metrics

- Accuracy
- Macro F1-Score (and weighted F1)
- Confusion Matrix
- Classification Report

---

## 📊 Results

### Model Performance Summary

| Model | Representation | Accuracy | Macro F1 |
|-------|---------------|----------|----------|
| Logistic Regression | BoW | ~0.66 | ~0.66 |
| Logistic Regression | TF-IDF | ~0.70 | ~0.70 |
| Naive Bayes | BoW | ~0.72 | ~0.72 |
| Naive Bayes | TF-IDF | ~0.74 | ~0.74 |
| Random Forest | BoW | ~0.60 | ~0.60 |
| Random Forest | TF-IDF | ~0.60 | ~0.60 |
| DNN | BoW | ~0.63 | ~0.63 |
| DNN | TF-IDF | ~0.66 | ~0.66 |
| DNN | GloVe | ~0.67 | ~0.66 |
| DNN | Skip-gram | ~0.70 | ~0.70 |
| GRU | GloVe (200d) | ~0.71 | ~0.71 |
| Bi-GRU | GloVe (200d) | ~0.71 | ~0.71 |
| GRU | Skip-gram (200d) | ~0.72 | ~0.72 |
| Bi-GRU | Skip-gram (200d) | **~0.7244** | **~0.72** |
| SimpleRNN | GloVe / Skip-gram | ~0.65–0.70 | ~0.64–0.70 |
| Bi-SimpleRNN | GloVe / Skip-gram | ~0.68–0.71 | ~0.68–0.71 |

### 🏆 Best Performing Model

**Bi-GRU + Skip-gram embeddings** achieved the highest accuracy of **72.44%**, outperforming all other configurations.

### 🔻 Worst Performing Model

**Random Forest + BoW/TF-IDF** showed the lowest performance (~60% accuracy), likely due to the high-dimensional sparse feature space and the model's inability to capture sequential dependencies.

### Key Observations

- Logistic Regression performed almost identically on BoW and TF-IDF for the test set.
- DNN with pre-trained embeddings (GloVe/Skip-gram) **failed to outperform** the best recurrent arrangement.
- Bi-directional RNN variants consistently outperformed their unidirectional counterparts.
- Confusion matrices revealed matching strengths across classes, indicating potential benefits from calibrated ensemble methods.

---

## 📈 Visualizations

The project includes the following visualizations (embedded in the notebook and report):

- Class distribution bar chart and pie chart
- Word cloud of most frequent terms
- Word count distribution by class (box plot)
- Correlation heatmap of text features
- Confusion matrices for all key models
- Accuracy/Loss curves (e.g., LSTM + GloVe)
- Model performance comparison bar charts

---

## 🛠️ Technologies & Libraries

- **Python 3.12**
- **scikit-learn** — TF-IDF, Logistic Regression, Multinomial Naive Bayes, Random Forest
- **TensorFlow / Keras** — DNN, SimpleRNN, GRU, LSTM, Bidirectional wrappers
- **Gensim** — Skip-gram Word2Vec training
- **NLTK** — Tokenization, stop-word removal, lemmatization
- **Matplotlib / Seaborn** — Visualization
- **WordCloud** — Word cloud generation
- **Pandas / NumPy** — Data manipulation

---

## 🚀 Getting Started

### Prerequisites

```bash
Install the necessary libraries if they are not installed on your local system. (See the libraries in the ipnyb file's first cell)
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the ' .ipynb ' files (sequentially) and run cells sequentially.

> ⚠️ **Note:** Ensure the dataset CSV files are placed in the `data/` directory before running.

---

## 📋 Deliverables

- [x] Jupyter Notebook (`.ipynb`) with well-organized sections, markdown explanations, and commented code
- [x] Project Report (IEEE Double-Column format, up to 6 pages)
- [x] All plots, tables, and evaluation results embedded in the notebook
- [x] Hyperparameter tuning experiments logged
- [x] Viva/Presentation preparation

---

## 🔮 Future Work

- Domain-adaptive pre-training for embeddings
- Contextual encoders (e.g., transformer-based models like BERT) with careful regularization
- Robust error analysis for minority phenomena
- Ensemble methods leveraging complementary model strengths

---

## 📚 References

1. scikit-learn `TfidfVectorizer` documentation
2. scikit-learn `LogisticRegression` (multinomial, solvers, regularization)
3. scikit-learn `MultinomialNB` (alpha smoothing)
4. Keras `SimpleRNN` layer API
5. Keras `GRU` layer API
6. Keras `Bidirectional` wrapper documentation

---

## 👥 Authors

- **Mohd Tashwaruddin Safin**
- **Fazle Mohammed Tasfiq**
- **Angshu Nirmegh Biswas**

Department of Computer Science and Engineering  
BRAC University, Dhaka, Bangladesh

---

## 📄 License

This project was completed as part of the **CSE440: Natural Language Processing II** Lab Project at BRAC University, Summer 2025.

---

## 🙏 Acknowledgments

- Course instructors for guidance
- BRAC University CSE Department for providing the dataset and infrastructure
