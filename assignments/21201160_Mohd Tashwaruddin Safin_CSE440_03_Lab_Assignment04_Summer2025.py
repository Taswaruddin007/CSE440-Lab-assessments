"""
Name: Mohd Tashwaruddin Safin  
ID: 21201160


# Lab Assignment - 04

Task: Token-Level Multi-Class Classification
"""

!pip install tensorflow

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
import nltk
from nltk.probability import FreqDist
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import random

nltk.download('punkt')

path = "/content/drive/MyDrive/CSE440_Project_dataset/assignment-04 dataset/Dataset_A_POS_train.csv"
pos = pd.read_csv(path)
pos.head()

pos.tail()

pos.info()

pos.dtypes

"""**Exploratory Data Analysis(EDA)**"""

print(f"Dataset Size: {len(pos)}\n")
print(f"Present of missing values:\n{pos.isnull().sum()}\n")

all_tags = [tag for tags in pos['POS'] for tag in eval(tags)]
unique_classes = set(all_tags)
print(f"Number of classes: {len(unique_classes)}")
print("The classes:", np.array(list(unique_classes)))

tag_counters = Counter(all_tags)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(tag_counters.keys()), y=list(tag_counters.values()))
plt.title("Class Distribution")
plt.xticks(rotation=90)
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

"""From the class distribution graph, it can be said that the dataset has imbalance class."""

# Text statistics
pos['sentence_length'] = pos['Sentence'].apply(lambda x: len(x.split()))
print(pos['sentence_length'].describe())

"""Here, Average sentence length is 21.840380"""

all_words = " ".join(pos['Sentence']).split()
fdist = FreqDist(all_words)
print(fdist.most_common(30))

top_words = fdist.most_common(30)

words = [w for w, _ in top_words]
counts = [c for _, c in top_words]

plt.figure(figsize=(12,6))
sns.barplot(x=counts, y=words, palette="viridis")
plt.title("Top 30 Most Frequent Words in the Dataset")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.show()

fdist.plot(30, cumulative=False)

"""**Data Preprocessing**"""

# Data Imbalance Check

all_tags = [tag for tags in pos['POS'] for tag in eval(tags)]
tag_counters = Counter(all_tags)

print(f"Tag Counts: {tag_counters}")
print(f"Most Common: {tag_counters.most_common(10)}")
print(f"Least Common: {tag_counters.most_common()[-10:]}")

plt.figure(figsize=(12, 6))
sns.barplot(x=list(tag_counters.keys()), y=list(tag_counters.values()), palette="viridis")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()

max_count = max(tag_counters.values())
min_count = min(tag_counters.values())
imbalance_ratio = max_count / min_count
print(f"Dataset Imbalance Ratio: {imbalance_ratio: .3f}")

"""From imbalance ratio, it can be said that the dataset is imbalance."""

all_tags = [tag for tags in pos["POS"] for tag in eval(tags)]
tag_counters = Counter(all_tags)

max_count = max(tag_counters.values())
scale = 0.5
target_count = int(max_count * scale)

new_sentences = []
new_tags = []

for sentence, tags in zip(pos['Sentence'], pos['POS']):
    tags_list = eval(tags)

    if any(tag_counters[tag] < target_count for tag in tags_list):
        min_freq = min(tag_counters[tag] for tag in tags_list)
        replicate_factor = int(np.ceil(target_count / min_freq))
    else:
        replicate_factor = 1

    for _ in range(replicate_factor):
        new_sentences.append(sentence)
        new_tags.append(tags)


upsampling_df = pd.DataFrame({'Sentence': new_sentences, 'POS': new_tags})

print(f"Original Dataset Size: {len(pos)}")
print(f"Upsampling Dataset Size: {len(upsampling_df)}")

all_tags_balanced = [tag for tags in upsampling_df["POS"] for tag in eval(tags)]
balanced_counts = Counter(all_tags_balanced)

print("Balanced class distribution:")
print(balanced_counts.most_common())

plt.figure(figsize=(12, 6))
sns.barplot(x=list(balanced_counts.keys()), y=list(balanced_counts.values()), palette="viridis")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()

all_tags_before = [tag for tags in pos["POS"] for tag in eval(tags)]
counts_before = Counter(all_tags_before)

all_tags_after = [tag for tags in upsampling_df["POS"] for tag in eval(tags)]
counts_after = Counter(all_tags_after)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(counts_before.keys(), counts_before.values())
axes[0].set_title("Class Distribution Before Upsampling")
axes[0].set_xlabel("Tags")
axes[0].set_ylabel("Frequency")
axes[0].tick_params(axis='x', rotation=90)

axes[1].bar(counts_after.keys(), counts_after.values(), color="orange")
axes[1].set_title("Class Distribution After Upsampling")
axes[1].set_xlabel("Tags")
axes[1].set_ylabel("Frequency")
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

# Tokenize and padding
sentences = upsampling_df['Sentence'].tolist()
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
X = tokenizer.texts_to_sequences(sentences)

upsampling_df['sentence_length'] = upsampling_df['Sentence'].apply(lambda x: len(x.split()))
max_len = max(upsampling_df['sentence_length'])
X = pad_sequences(X, maxlen=max_len, padding='post')

tag_encoder = LabelEncoder()
all_tags = [tag for tags in upsampling_df['POS'] for tag in eval(tags)]
all_tags.append('O')
tag_encoder.fit(all_tags)

y = []
for tags in upsampling_df['POS']:
    encoded = tag_encoder.transform(eval(tags))
    padded = np.pad(encoded, (0, max_len-len(encoded)), "constant", constant_values=tag_encoder.transform(["O"])[0])
    y.append(padded)

y = np.array(y)

# Simple RNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

vocab_size = len(tokenizer.word_index) + 1
num_tags = len(tag_encoder.classes_)

with tf.device('/GPU:0'):
    rnn_model = Sequential(
        [Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len, mask_zero=True),
         SimpleRNN(128, return_sequences=True),
         Dropout(0.3),
         Dense(num_tags, activation='softmax')
         ]
    )

    rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history_rnn = rnn_model.fit(
        X, y,
        validation_split=0.2,
        batch_size=32,
        epochs=15
        )

"""Hyperparameters Description:

   input_dim=vocab_size: This parameter is set to ensure that the embedding table has one row for each word or token in the vocabulary.
  
   output_dim=128: This represents the embedding size, or the vector dimension for each token. For a dataset (after upsampling), 128 is suitable for a large dataset. It's not too small (like 16 or 32, which may lead to underfitting).

   input_length=max_len: This ensures that the embeddings align with the padded input sequences.

   mask_zero=True: This allows the RNN to ignore the padding (index 0), preventing it from wasting capacity on meaningless padding tokens.

   Embedding Size and Hidden Units: The choice of 128 hidden units matches the embedding dimension, providing symmetry between the input features and the hidden representation.

   return_sequences=True: This is essential for sequence labeling tasks, such as Part of Speech (POS) tagging. It enables the model to return predictions for each time step (each word) instead of just the last hidden state.

   Dropout Rate = 0.3: This randomly drops 30% of the neurons during training, helping to prevent overfitting. This is particularly important since upsampling may lead to duplicated examples. Without dropout, the model might simply memorize repeated sentences. A dropout rate of 0.3 is a commonly accepted trade-off:
    Too low (<0.2) may not provide sufficient regularization.
    Too high (>0.5) may lead to underfitting.

   num_tags: This indicates the number of POS tags (classes) to predict.

   Optimizer: Adam: Adam uses an adaptive learning rate, making it well-suited for NLP tasks that involve sparse gradients (like embeddings). The default learning rate of 0.001 is a good starting point for RNN-based models, chosen for its potential to ensure fast convergence and stability on sequence tasks.

   Activation Function: Softmax: Softmax produces a probability distribution over all tags for each token, making it standard for multi-class classification. Since the labels are integer encoded (with integers representing each tag), using sparse_categorical_crossentropy is appropriate. This approach avoids the need for one-hot encoding, making it more memory efficient for sequence tasks.

   validation_split=0.2: This reserves 20% of the data for validation, which helps monitor the model's generalization capabilities.

   batch_size=32: This size is balanced; it's small enough to capture gradient updates frequently and large enough to ensure GPU efficiency, especially since the dataset was enlarged due to upsampling.

   epochs=15: This epoch count is slightly higher than the previous setting of 12 due to the increased data from upsampling. It allows the model to observe all examples (both replicated and minority instances) sufficiently to learn patterns effectively.

This configuration has been chosen to maintain a balance between training stability and model performance on the upsampled dataset.

"""

rnn_model.summary()

# Simple RNN Model
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history_rnn.history['loss'], label='Training Loss')
plt.plot(history_rnn.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history_rnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_rnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# LSTM
from tensorflow.keras.layers import LSTM

with tf.device('/GPU:0'):
    lstm_model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len, mask_zero=True),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        Dense(num_tags, activation="softmax")
    ])

    lstm_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history_lstm = lstm_model.fit(
        X, y,
        validation_split=0.2,
        batch_size=32,
        epochs=15
    )

"""Hyperparameters description:

   input_dim=vocab_size: This parameter defines how many words or tokens can be represented, indicating the size of your vocabulary.

   output_dim=128: The embedding dimension is set to 128, which allows LSTMs can utilize richer embeddings, enhancing expressiveness, especially since POS tagging involves subtle syntactic distinctions.

   mask_zero=True: This setting ensures that padding tokens (0) are ignored during training, which is crucial for sequence labeling tasks. Aiming for richer semantic representation without excessively high dimensions (like 256 or 300), which could lead to overfitting.

   128 units: This refers to the number of memory cells (hidden units). LSTMs are more powerful than SimpleRNNs, and having 128 units allows them to learn longer dependencies in sentences, such as subject–verb agreement and context across multiple words.

   return_sequences=True: This parameter is essential for sequence labeling (like POS tagging) because it returns output for each timestep instead of just the last one. LSTMs excel at capturing long-term dependencies that traditional RNNs struggle with.

   Dropout rate = 0.3: This means that 30% of neurons are randomly disabled during training to help reduce overfitting, particularly since your dataset has been upsampled, which carries a risk of memorization. A moderate dropout value (0.3) provides regularization without significantly hindering learning. Regularization is used to prevent the LSTM from memorizing duplicate sentences.

   num_tags: This denotes the number of POS tags (classes).

   Softmax: This function converts outputs into class probability distributions for each token. Multi-class sequence classification.

   Adam: This optimizer adapts learning rates for each parameter and works well with LSTMs, as it effectively handles sparse gradients, which are common in NLP tasks, and converges faster than plain SGD. Fast convergence and robustness in sequence learning.

   Labels (y): These are integers (not one-hot encoded), so using sparse categorical crossentropy is memory efficient and appropriate. This is a standard choice for multi-class classification with integer-encoded labels.

   batch_size=32: This size is small enough to ensure stable gradient updates while being large enough for GPU parallelism.

   epochs=15: This number of epochs provides the LSTM with sufficient passes over the upsampled data to learn dependencies. Remember that LSTMs require more time to converge compared to RNNs.

   validation_split=0.2: This reserves 20% of the training set for validation to monitor for overfitting.

A balanced approach between training efficiency and performance monitoring.

"""

lstm_model.summary()

# LSTM Model
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history_lstm.history['loss'], label='Training Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history_lstm.history['accuracy'], label='Training Accuracy')
plt.plot(history_lstm.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

"""**Test Set implementation**"""

path2 = "/content/drive/MyDrive/CSE440_Project_dataset/assignment-04 dataset/Dataset_A_POS_test.csv"
test = pd.read_csv(path2)
test.head()

# test tokenizer and padding
sentences_test = test['Sentence'].tolist()
X_test = tokenizer.texts_to_sequences(sentences_test)
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

y_test = []
for tags in test['POS']:
    encoded = tag_encoder.transform(eval(tags))
    padded = np.pad(encoded, (0, max_len - len(encoded)),
                    "constant", constant_values=tag_encoder.transform(["O"])[0])
    y_test.append(padded)

y_test = np.array(y_test)

# Simple RNN model evaluation
rnn_test_loss, rnn_test_acc = rnn_model.evaluate(X_test, y_test)
print(f"Simple RNN Test Loss: {rnn_test_loss: .3f}")
print(f"Simple RNN Test Accuracy: {rnn_test_acc: .3f}")

metrics = ['Test Loss', 'Test Accuracy']
values = [rnn_test_loss, rnn_test_acc]

plt.bar(metrics, values, color=['red', 'green'])
plt.title("Simple RNN Model Performance on Test Set")
plt.ylabel("Value")
plt.show()

# LSTM Model Evaluation

lstm_test_loss, lstm_test_acc = lstm_model.evaluate(X_test, y_test)
print(f"LSTM Test Loss: {lstm_test_loss: .3f}")
print(f"LSTM Test Accuracy: {lstm_test_acc: .3f}")

metrics = ['Test Loss', 'Test Accuracy']
values = [lstm_test_loss, lstm_test_acc]

plt.bar(metrics, values, color=['red', 'green'])
plt.title("LSTM Model Performance on Test Set")
plt.ylabel("Value")
plt.show()

# Classification Report

from sklearn.metrics import classification_report, confusion_matrix

y_pred_rnn = rnn_model.predict(X_test)
y_pred_rnn = np.argmax(y_pred_rnn, axis=-1)

y_pred_lstm = lstm_model.predict(X_test)
y_pred_lstm = np.argmax(y_pred_lstm, axis=-1)

y_true_flat = y_test.flatten()
y_pred_rnn_flat = y_pred_rnn.flatten()
y_pred_lstm_flat = y_pred_lstm.flatten()

print("Simple RNN Classification Report")
print(classification_report(y_true_flat, y_pred_rnn_flat, target_names=tag_encoder.classes_))
print()
print("LSTM Classification Report")
print(classification_report(y_true_flat, y_pred_lstm_flat, target_names=tag_encoder.classes_))

# Confusion Matrix

cm = confusion_matrix(y_true_flat, y_pred_rnn_flat)
plt.figure(figsize=(12,8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix - Simple RNN")
plt.show()

cm = confusion_matrix(y_true_flat, y_pred_lstm_flat)
plt.figure(figsize=(12,8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix - LSTM")
plt.show()

"""**1. Accuracy**  
The Simple RNN demonstrated an accuracy of 67%, significantly outperforming the LSTM, which achieved an accuracy of 41%.  

**2. Precision/Recall (Macro Average)**  
For the Simple RNN:  
- Precision = 0.07  
- Recall = 0.03  
- F1 Score = 0.03  

For the LSTM:  
- Precision = 0.03  
- Recall = 0.03  
- F1 Score = 0.02  

Both models exhibited challenges related to class imbalance, as evidenced by low macro averages. However, the Simple RNN maintained a slight advantage.

**3. Confusion Matrix Observation**  
The confusion matrices for both models indicate a dominance of predictions for a limited number of classes, likely corresponding to the most frequent tags, while predictions for the remaining classes are sparse.  
- The Simple RNN confusion matrix reveals a concentrated prediction density around a small cluster, suggesting some overfitting to frequent classes, although it displays slightly more distribution compared to the LSTM.  
- The LSTM's confusion matrix reflects an even more pronounced focus on a few classes, signifying a tendency for the model to predict predominantly the dominant categories.

**Conclusion**  
In summary, the Simple RNN outperformed the LSTM in this analysis. This outcome is likely attributable to the interplay between model complexity and dataset characteristics. The RNN's simpler architecture enabled it to effectively learn frequent patterns, while the LSTM may have overfit or focused excessively on majority classes due to its higher parameter count and a potential lack of sufficient data for less frequent classes.

"""

