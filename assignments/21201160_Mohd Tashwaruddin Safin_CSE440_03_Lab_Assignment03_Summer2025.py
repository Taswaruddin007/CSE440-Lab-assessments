
"""

Name: Mohd Tashwaruddin Safin

ID: 21201160

# Lab Assignment 03

**Task-01**
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

dataset = load_iris()
dataset.keys()

X = dataset['data']
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# As the dataset has 3 class, so it encode the classes in one-hot encoding
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Neural Network Model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1], )))
model.add(Dense(3, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

training_data = model.fit(
    X_train, y_train_categorical, validation_split=0.2,
    epochs=200, verbose=1, batch_size=15
)

print("Model Summary:")
model.summary()

loss, accuracy = model.evaluate(X_test, y_test_categorical, verbose=1)
print(f"Test Loss: {loss*100: .3f} %")
print(f"Test Accuracy: {accuracy*100: .3f} %")

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(training_data.history['accuracy'], label='Train Accuracy')
plt.plot(training_data.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy VS Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_data.history['loss'], label='Train Loss')
plt.plot(training_data.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss VS Epochs')
plt.legend()

plt.show()

"""**Task-02**"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

data = imdb.load_data(num_words=10000)
(X_train_full, y_train_full), (X_test, y_test) = data

max_len = 250
X_train_full = pad_sequences(X_train_full, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64, input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train, epochs=12,
    batch_size=128, validation_data=(X_val, y_val),
    verbose=1
)

print('Model Summary: ')
model.summary()

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f'Test Loss: {test_loss*100: .3f} %')
print(f'Test Accuracy: {test_accuracy*100: .3f} %')

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy VS Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss VS Epochs')
plt.legend()

plt.show()

y_pred = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print(f"AUC(area under the curve) takes: {roc_auc*100 : .3f} %")

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc: .3f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

"""**Task-03**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = imdb.load_data(num_words=10000)
(X_train_full, y_train_full), (X_test, y_test) = data

max_len = 250
X_train_full = pad_sequences(X_train_full, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

def build_shallow_model():
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=64, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )

    return model

def build_deep_model():
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=64, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'), Dropout(0.2))
    model.add(Dense(32, activation='relu'), Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

shallow_model = build_shallow_model()
start_time = time.time()
history = shallow_model.fit(X_train, y_train, epochs=12, batch_size=128, validation_data=(X_val, y_val),
                            verbose=1)
finish_time = time.time()
shallow_training_time = finish_time - start_time

shallow_test_loss, shallow_test_accuracy = shallow_model.evaluate(X_test, y_test, verbose=1)

print(f"Shallow Model Test Loss: {shallow_test_loss*100: .3f} %")
print(f"Shallow Model Test Accuracy: {shallow_test_accuracy*100: .3f} %")

deep_model = build_deep_model()
start_time = time.time()
history_deep = deep_model.fit(X_train, y_train, epochs=18, batch_size=128,
                              validation_data=(X_val, y_val), verbose=1)
finish_time = time.time()
deep_training_time = finish_time - start_time

deep_test_loss, deep_test_accuracy = deep_model.evaluate(X_test, y_test, verbose=1)

print(f"Deep Model Test Loss: {deep_test_loss*100: .3f} %")
print(f"Deep Model Test Accuracy: {deep_test_accuracy*100: .3f} %")

# For shallow model
y_pred = (shallow_model.predict(X_test) >= 0.5).astype('int32')
conf_mat = confusion_matrix(y_test, y_pred)
conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Negative', 'Positive'])
conf_mat_disp.plot(cmap=plt.cm.Blues)
plt.title("Shallow Model Confusion Matrix")
plt.show()

# For deep model
y_pred = (deep_model.predict(X_test) >= 0.5).astype('int32')
conf_mat = confusion_matrix(y_test, y_pred)
conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Negative', 'Positive'])
conf_mat_disp.plot(cmap=plt.cm.Blues)
plt.title("Deep Model Confusion Matrix")
plt.show()

print("---Comparison Between Shallow and Deep Model---\n")
print("For Shallow Model:")
print(f"Shallow Model Training Time: {shallow_training_time: .3f} seconds, Test Accuracy: {shallow_test_accuracy*100: .3f} %")
print(f"Overfitting Gap = {history.history['accuracy'][-1] - history.history['val_accuracy'][-1]: .3f}")
print("\nFor Deep Model:")
print(f"Deep Model Training Time: {deep_training_time:.3f} seconds, Test Accuracy: {deep_test_accuracy*100: .3f} %")
print(f"Overfitting Gap = {history_deep.history['accuracy'][-1] - history_deep.history['val_accuracy'][-1]: .3f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Shallow Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Shallow Validation Accuracy')
plt.plot(history_deep.history['accuracy'], label='Deep Train Accuracy')
plt.plot(history_deep.history['val_accuracy'], label='Deep Validation Accuracy')
plt.title('Accuracy Comparison')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Shallow Train Loss')
plt.plot(history.history['val_loss'], label='Shallow Validation Loss')
plt.plot(history_deep.history['loss'], label='Deep Train Loss')
plt.plot(history_deep.history['val_loss'], label='Deep Validation Loss')
plt.title('Loss Comparison')
plt.legend()

"""The presented plots offer a comparative analysis of training and validation accuracy and loss for both shallow and deep models applied to IMDb sentiment analysis.

In the Accuracy Comparison (left), it is observed that both models achieve high training accuracy relatively quickly. However, the deep model, represented in green, attains an accuracy level nearing 100% within a few epochs, while its validation accuracy, indicated in red, stabilizes around 85% and exhibits a plateau—this pattern is indicative of significant overfitting. Conversely, the shallow model, depicted in blue and orange, demonstrates a narrower gap between training and validation accuracy, signifying superior generalization capabilities despite not achieving perfect training accuracy.

In the Loss Comparison (right), the training loss for the deep model (green) approaches near-zero values; however, its validation loss (red) shows a consistent increase after the initial epochs, further corroborating the presence of overfitting. The validation loss for the shallow model (orange) remains more stable and closely aligns with the training loss (blue), suggesting an effective equilibrium between fitting the training data and generalizing to new instances.

In conclusion, the visual evidence presented indicates that while the deep model tends to memorize the training dataset, the shallow model exhibits improved generalization capabilities. So, the shallow model performs better overall.

**Task-04**
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from sklearn.metrics.pairwise import cosine_similarity

num_words = 10000
(X_train_full, y_train_full), (X_test, y_test) = imdb.load_data(num_words=num_words)

def binary_bow(sequence, dimension=num_words):
    results = np.zeros((len(sequence), dimension))
    for i, seq in enumerate(sequence):
        results[i, seq] = 1
    return results

X_train = binary_bow(X_train_full)
X_test = binary_bow(X_test)

X_val = X_train[:10000]
partial_x_train = X_train[10000:]

#Encoder
input_layer = Input(shape=(num_words,))
encoded = Dense(512, activation='relu')(input_layer)
encoded = Dense(128, activation='relu')(encoded)

#Decoder
decoded = Dense(512, activation='relu')(encoded)
output_layer = Dense(num_words, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer)

autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = autoencoder.fit(
    partial_x_train, partial_x_train,
    epochs=25,
    batch_size=128,
    validation_data=(X_val, X_val)
)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train & Validation Loss Curves')
plt.legend()
plt.show()

reconstructed_vector = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - reconstructed_vector))
print(f"Reconstruction Error using MSE on test set: {reconstruction_error: .3f}")

cos_sim = []
for i in range(len(X_test)):
    sim = cosine_similarity(
        X_test[i].reshape(1, -1),
        reconstructed_vector[i].reshape(1, -1),
    )[0, 0]
    cos_sim.append(sim)

avg_cos_sim = np.mean(cos_sim)
print(f"Average cosine similarity between the original and reconstructed vector is {avg_cos_sim: .3f}")

import seaborn as sns

n_samples = 20
subset_original = X_test[:n_samples]
subset_reconstructed = reconstructed_vector[:n_samples]

cos_sim_matrix = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        cos_sim_matrix[i, j] = cosine_similarity(
            subset_original[i].reshape(1, -1),
            subset_reconstructed[j].reshape(1, -1)
        )[0,0]

plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_matrix, cmap="viridis", annot=True)
plt.xlabel("Reconstructed Samples")
plt.ylabel("Original Samples")
plt.title("Cosine Similarity Heatmap (Original vs Reconstructed)")
plt.show()

"""The cosine similarity heatmap offers valuable insights into the performance of the autoencoder applied to the IMDb dataset. The diagonal values, reflecting the similarity between each original sample and its corresponding reconstruction, predominantly range from 0.55 to 0.65. This range suggests that the autoencoder captures certain significant features of the input data; however, it does not achieve a high level of reconstruction accuracy, as ideal values would approach 1.

In contrast, the off-diagonal values, generally falling between 0.25 and 0.45, are significantly lower, indicating that the reconstructions are more similar to their respective inputs than to other samples. This observation is a positive indicator, as it implies that the model is effectively learning sample-specific representations as opposed to generating generic outputs.

Nevertheless, the relatively modest similarity scores observed on the diagonal imply that much of the intricate details may be lost during the compression process. This limitation is likely a consequence of the challenges associated with reconstructing sparse, high-dimensional multi-hot vectors within a relatively constrained latent space.

Overall, the autoencoder demonstrates a degree of success in distinguishing inputs and preserving certain structural elements; however, its reconstruction capabilities appear to be limited.
"""