"""Leaky: a text classifier with two separate leaks before cross-validation.

TF-IDF learns its vocabulary and weights from every document, and chi2 selects terms
using every label. Selection usually inflates the score far more than the vocabulary does.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

reviews = pd.read_csv("reviews.csv")
texts = reviews["text"]
labels = reviews["sentiment"]

vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(texts)
X_best = SelectKBest(chi2, k=2000).fit_transform(X_tfidf, labels)  # foldsafe-expect: FS002

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(LogisticRegression(max_iter=1000), X_best, labels, cv=cv)  # foldsafe-expect: FS003
print(f"F1: {scores.mean():.3f}")
