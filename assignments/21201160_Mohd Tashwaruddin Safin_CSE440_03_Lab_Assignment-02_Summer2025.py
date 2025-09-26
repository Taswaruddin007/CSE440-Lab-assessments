
"""

Name: Mohd Tashwaruddin Safin
ID: 21201160

# Lab Assignment 02

**Task-01**
"""

import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('all')

path = "/content/IMDB Dataset.csv"
reviews = pd.read_csv(path)

reviews['label'] = reviews['sentiment'].map({'positive':1, 'negative':0})

def process_reviews(text):
    text = text.lower()
    tokens = word_tokenize(text)
    remove_punc = [word for word in tokens if word.isalnum()]
    stopword = stopwords.words('english')
    remove_stopwords = [word for word in remove_punc if word not in stopword]
    return ' '.join(remove_stopwords)

reviews['clean_review'] = reviews['review'].apply(process_reviews)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews['clean_review'])
y = reviews['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)
classification = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])

classes = ["Negative", "Positive"]
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=classes, yticklabels=classes)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

print(f"Macro-F1 score: {macro_f1 : .3f}\n")
print(f"Confusion Matrix: \n{conf_matrix}\n")
print(f"Classification Report: \n{classification}")

"""**Task-02**"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def glove_embedding_load(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

path = '/content/glove.6B.100d.txt'
glove_embeddings = glove_embedding_load(path)

def get_word_vector(word):
    vector = glove_embeddings[word.lower()]
    return vector

teacher_vec = get_word_vector('Teacher')
educate_vec = get_word_vector('Educate')
heal_vec = get_word_vector('Heal')

if(teacher_vec is None or educate_vec is None or heal_vec is None):
    raise ValueError('One or more words not found in the embeddings.')

result_vec = teacher_vec - educate_vec + heal_vec

def find_closest(result_vec, embeddings):
    words = []
    vectors = []
    for word, vector in embeddings.items():
        words.append(word)
        vectors.append(vector)
    vectors = np.array(vectors)

    similarity = cosine_similarity([result_vec], vectors).reshape(-1)
    sorted_indices = np.argsort(similarity)[::-1]

    result = [(words[i], similarity[i]) for i in sorted_indices if words[i].lower() not in ['teacher', 'educate', 'heal']]
    return result

closest_words = find_closest(result_vec, glove_embeddings)

print('Top 10 Closest Words for: (teacher - educate + heal) is --')
for word, value in closest_words[:10]:
    print(f'{word} : {value: .5f}')

flag = None
if(closest_words[0][0].lower() == 'doctor'):
    flag = 'Yes'
else:
    flag = 'No'

print('\nIs "Doctor" the closest word of "Teacher - Educate + Heal"? -->', flag)

"""**Task-03**"""

!pip install gensim

import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import brown
from nltk import word_tokenize
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from nltk.probability import FreqDist
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('all')

sentences = brown.sents() # tokenize

# removes punctuations and lower the word
process_sents = [
    [word.lower() for word in sent if word.isalnum()]
    for sent in sentences
]

#CBOW model
cbow = Word2Vec(
    sentences=process_sents,
    vector_size=150,
    window=4,
    min_count=2,
    sg=0
)


#Skip-gram model
skipgram = Word2Vec(
    sentences=process_sents,
    vector_size=150,
    window=4,
    min_count=2,
    sg=1
)

print(f"CBOW model word similarity (teacher and educate): {cbow.wv.similarity('teacher', 'educate')}")
print(f"CBOW model word similarity (doctor and heal): {cbow.wv.similarity('doctor', 'heal')}\n")

print(f"Skip-gram model word similarity (teacher and educate): {skipgram.wv.similarity('teacher', 'educate')}")
print(f"Skip-gram model word similarity (doctor and heal): {skipgram.wv.similarity('doctor', 'heal')}")

print(f"CBOW model most similarity with 'teacher' is {cbow.wv.most_similar('teacher', topn=3)}")
print(f"Skip-gram model most similarity with 'teacher' is {skipgram.wv.most_similar('teacher', topn=3)}\n")

print(f"COBW model most similarity with 'doctor' is {cbow.wv.most_similar('doctor', topn=3)}")
print(f"Skip-gram model most similarity with 'doctor' is {skipgram.wv.most_similar('doctor', topn=3)}")

# for cbow
teacher_vector = cbow.wv['teacher']
educate_vector = cbow.wv['educate']
heal_vector = cbow.wv['heal']
doctor_vector = cbow.wv['doctor']

result = teacher_vector - educate_vector + heal_vector
similarity = cosine_similarity([result], [doctor_vector])
print(f"CBOW model cosine similarity for teacher - educate + heal with doctor is {similarity[0][0]: .3f}")

# for skipgram
teacher_vector = skipgram.wv['teacher']
educate_vector = skipgram.wv['educate']
heal_vector = skipgram.wv['heal']
doctor_vector = skipgram.wv['doctor']

result = teacher_vector - educate_vector + heal_vector
similarity = cosine_similarity([result], [doctor_vector])
print(f"Skip-gram model cosine similarity for teacher - educate + heal with doctor is {similarity[0][0]: .3f}")

def analogy(model, w1, w2, w3):
    result = model.wv.most_similar(positive=[w1, w3], negative=[w2], topn=1)
    return result


print(f"CBOW analogy: {analogy(cbow, 'teacher', 'educate', 'heal')}")
print(f"Skip-gram analogy: {analogy(skipgram, 'teacher', 'educate', 'heal')}")

all_words = [word for sent in process_sents for word in sent]
fdist = FreqDist(all_words)
most_freq_50 = [w for w, _ in fdist.most_common(50) if w in cbow.wv and w in skipgram.wv]

pca = PCA(n_components=2)

cbow_vectors = np.array([cbow.wv[word] for word in most_freq_50])
cbow_2d = pca.fit_transform(cbow_vectors)

skipgram_vectors = np.array([skipgram.wv[word] for word in most_freq_50])
skipgram_2d = pca.fit_transform(skipgram_vectors)

def plot_embeddings(words, coords, title, ax):
    ax.scatter(coords[:, 0], coords[:, 1])
    for i, word in enumerate(words):
        ax.annotate(word, xy=(coords[i, 0], coords[i, 1]))
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_embeddings(most_freq_50, cbow_2d, 'CBOW Embeddings', axes[0])
plot_embeddings(most_freq_50, skipgram_2d, 'Skip-gram Embeddings', axes[1])

"""The provided plots illustrate a two-dimensional Principal Component Analysis (PCA) projection of the Word2Vec embedding vectors for the 50 most frequent terms in the Brown corpus, generated using both the Continuous Bag of Words (CBOW) and Skip-gram architectures.

In the case of the CBOW model, the embeddings demonstrate loosely defined groupings, indicative of the model's propensity to capture broader semantic relationships through frequent contextual patterns. This observation aligns with the intrinsic characteristic of CBOW, which averages contextual information, resulting in more generalized word representations.

In contrast, the Skip-gram model yields more distinct and compact clusters, suggesting an enhanced ability to capture specific semantic and syntactic relationships among words. This phenomenon can be attributed to the Skip-gram training objective, which emphasizes predicting context words based on a given target, thereby facilitating richer representations, particularly for less frequent terms.

In summary, the visualizations corroborate the empirical understanding that, while CBOW is computationally efficient and effective for modeling frequent words, the Skip-gram approach produces embeddings of superior quality characterized by greater semantic granularity.
"""