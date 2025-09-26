import nltk
nltk.download("all")

"""**Task-01**"""

nltk.download("brown")
nltk.download("reuters")

from nltk.corpus import brown

print(f"Brown corpus available categories: {brown.categories()}")
print(f"Brown corpus total categories: {len(brown.categories())}")

from nltk.corpus import reuters

print(f"Reuters corpus available text files: {reuters.fileids()}")
print(f"Reuters corpus total text files: {len(reuters.fileids())}")

"""**Task-02**"""

nltk.download("gutenberg")

import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

file1 = "austen-sense.txt"
file2 = "edgeworth-parents.txt"

def processing_text(file):
    text = gutenberg.raw(file)
    tokens = word_tokenize(text.lower())
    return tokens

file1_tokens = processing_text(file1)
file2_tokens = processing_text(file2)

print(f"file1 tokens: {file1_tokens}\n")
print(f"file2 tokens: {file2_tokens}")

def remove_punctuation_stopwords(tokens):
    filtered_tokens = [word for word in tokens if word.isalpha()] #remove punctuations
    stopword = set(stopwords.words("english"))
    filtered_tokens = [word for word in filtered_tokens if word not in stopword] #remove stopwords
    return " ".join(filtered_tokens)

final_text1 = remove_punctuation_stopwords(file1_tokens)
final_text2 = remove_punctuation_stopwords(file2_tokens)

wc1 = WordCloud(width=800, height=400, background_color="white").generate(final_text1)
wc2 = WordCloud(width=800, height=400, background_color="white").generate(final_text2)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(wc1, interpolation="bilinear")
plt.axis("off")
plt.title(f"{file1}")

plt.subplot(1, 2, 2)
plt.imshow(wc2, interpolation="bilinear")
plt.axis("off")
plt.title(f"{file2}")

plt.show()

"""**Task-03**"""

nltk.download("brown")

import nltk
from nltk.corpus import brown, stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
import matplotlib.pyplot as plt

fictional_text = brown.words(categories="fiction")
text = " ".join(fictional_text).lower()

tokens = word_tokenize(text)
words = [word for word in tokens if word.isalpha()]

stopword = set(stopwords.words("english"))
filtered_words = [word for word in words if word not in stopword]

def pos_tag(tag):
    if(tag.startswith("J")):
        return wordnet.ADJ
    elif(tag.startswith("V")):
        return wordnet.VERB
    elif(tag.startswith("N")):
        return wordnet.NOUN
    elif(tag.startswith("R")):
        return wordnet.ADV
    else:
        return wordnet.NOUN

pos_tags = nltk.pos_tag(filtered_words)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word, pos_tag(pos)) for word, pos in pos_tags]

print(f"Tokenization: {words}")
print(f"Removing stopwords: {filtered_words}")
print(f"Lemmatization: {lemmatized_words}")

cleaned_text = lemmatized_words
fdist = FreqDist(cleaned_text)
most_common_words = fdist.most_common(15)

print(f"Most frequent words: {most_common_words}")

words = [word for word, freq in most_common_words]
freq = [freq for word, freq in most_common_words]
plt.figure(figsize=(12, 6))
plt.bar(words, freq, color="blue")
plt.title("15 Most Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=80)
plt.show()

"""**Task-04**"""

nltk.download("gutenberg")

import nltk
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from nltk import FreqDist
import pandas as pd

text = gutenberg.raw("carroll-alice.txt").lower()

tokens = word_tokenize(text)
words = [word for word in tokens if word.isalpha()]

stopword = set(stopwords.words("english"))
filtered_words = [word for word in words if word not in stopword]

print(f"Tokenization & punctuation remove: {words}")
print(f"Stopwords remove: {filtered_words}")

fdist = FreqDist(filtered_words)
top20_words = fdist.most_common(20)
top20_words1 = [word for word, freq in top20_words]
print(f"Top 20 frequent words: {top20_words1}")

bigram_list = list(bigrams(filtered_words))
print(f"Bigrams list: {bigram_list}")

co_occurance = pd.DataFrame(0, index=top20_words1, columns=top20_words1)

for word1, word2 in bigram_list:
    if(word1 in top20_words1 and word2 in top20_words1):
        co_occurance.loc[word1, word2] += 1

print(f"Co-occurance Matrix for Top 20 Frequent Words:")
print(co_occurance)

"""**Task-05**"""

nltk.download("udhr")

import nltk
from nltk.corpus import udhr
from nltk.util import ngrams
from nltk import FreqDist

hungarian_words = udhr.words("Hungarian_Magyar-UTF8")

vowels = "aáeéiíoóöőuúüű"

def extract_vowels(word):
    val = [i for i in word.lower() if i in vowels]
    return "".join(val)

all_trigrams = []

for word in hungarian_words:
    extracted_vowels = extract_vowels(word)
    if(len(extracted_vowels) >= 3):
        trigrams = ngrams(extracted_vowels, 3)
        t1 = ["".join(i) for i in trigrams]
        all_trigrams += t1


trigram_freq = FreqDist(all_trigrams)

sorted_trigram = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)

print("Sorted List of Trigrams and its Count: ")
for trigram, freq in sorted_trigram:
    print(f"{trigram} : {freq}")