"""Leaky: PCA components are learned from all rows in a one-line chained call."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X = np.load("features.npy")
y = np.load("labels.npy")

X_reduced = PCA(n_components=10).fit_transform(X)  # foldsafe-expect: FS001
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, stratify=y, random_state=1)
print(SVC().fit(X_train, y_train).score(X_test, y_test))
