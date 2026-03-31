# ==============================
# IMPORTS
# ==============================
import pandas as pd
import numpy as np
import time
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

nltk.download('stopwords')


# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("spam.csv", encoding='latin1')

# Keep only needed columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'content']

# Drop nulls
df = df.dropna()


# ==============================
# CLEAN TEXT
# ==============================
stop_words = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    return " ".join(words)   # (no stopword removal for better results)

df['clean'] = df['content'].apply(clean)


# ==============================
# LABEL CONVERSION
# ==============================
df['label'] = df['label'].map({'ham':0, 'spam':1})

print("Label Distribution:\n", df['label'].value_counts())


# ==============================
# SPLIT DATA
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42
)


# ==============================
# EVALUATION FUNCTION
# ==============================
def evaluate(y_test, pred):
    return (
        accuracy_score(y_test, pred),
        precision_score(y_test, pred),
        recall_score(y_test, pred),
        f1_score(y_test, pred)
    )


results = []


# ==============================
# METHOD 1: BoW
# ==============================
start = time.time()

bow = CountVectorizer(max_features=5000)
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_bow, y_train)

pred = model.predict(X_test_bow)

results.append(["BoW", *evaluate(y_test, pred), time.time() - start])


# ==============================
# METHOD 2: TF-IDF
# ==============================
start = time.time()

tfidf = TfidfVectorizer(max_features=5000)
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_tf, y_train)

pred = model.predict(X_test_tf)

results.append(["TF-IDF", *evaluate(y_test, pred), time.time() - start])


# ==============================
# METHOD 3: N-GRAMS
# ==============================
start = time.time()

ng = CountVectorizer(ngram_range=(1,2), max_features=5000)
X_train_ng = ng.fit_transform(X_train)
X_test_ng = ng.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_ng, y_train)

pred = model.predict(X_test_ng)

results.append(["N-grams", *evaluate(y_test, pred), time.time() - start])


# ==============================
# METHOD 4: WORD2VEC
# ==============================
sentences = [text.split() for text in X_train]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=2)

def avg_vec(text):
    words = text.split()
    vecs = [w2v.wv[w] for w in words if w in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

start = time.time()

X_train_w2v = np.array([avg_vec(text) for text in X_train])
X_test_w2v = np.array([avg_vec(text) for text in X_test])

model = LogisticRegression(max_iter=2000)
model.fit(X_train_w2v, y_train)

pred = model.predict(X_test_w2v)

results.append(["Word2Vec", *evaluate(y_test, pred), time.time() - start])


# ==============================
# FINAL RESULTS
# ==============================
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score", "Time"
])

print("\nFINAL RESULTS:\n")
print(results_df)


# ==============================
# PLOTS
# ==============================
import matplotlib.pyplot as plt

results_df.plot(x="Model", y="Accuracy", kind="bar", title="Accuracy Comparison")
plt.show()

results_df.plot(x="Model", y="Time", kind="bar", title="Time Comparison")
plt.show()