#!/usr/bin/env python3
"""
Tabular baseline models for function-level multilabel classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class _ConstantLabelModel:
    value: int

    def predict_proba(self, features: spmatrix) -> np.ndarray:
        count = features.shape[0]
        proba = np.zeros((count, 2), dtype=np.float32)
        proba[:, self.value] = 1.0
        return proba


class TabularMultilabelBaseline:
    """
    TF-IDF + one-vs-rest logistic regression baseline.

    Each label is trained independently so rare labels with constant training targets
    can fall back to a constant predictor rather than crashing the run.
    """

    def __init__(
        self,
        *,
        max_features: int = 50000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: tuple[int, int] = (1, 2),
        c_value: float = 4.0,
        max_iter: int = 1000,
    ):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
        )
        self.c_value = c_value
        self.max_iter = max_iter
        self.models: list[_ConstantLabelModel | LogisticRegression] = []

    def fit(self, texts: list[str], labels: np.ndarray):
        features = self.vectorizer.fit_transform(texts)
        self.models = []
        for idx in range(labels.shape[1]):
            y = labels[:, idx]
            unique = np.unique(y)
            if unique.size < 2:
                self.models.append(_ConstantLabelModel(int(unique[0]) if unique.size else 0))
                continue

            model = LogisticRegression(
                C=self.c_value,
                max_iter=self.max_iter,
                solver="liblinear",
                class_weight="balanced",
            )
            model.fit(features, y)
            self.models.append(model)
        return self

    def transform(self, texts: list[str]) -> spmatrix:
        return self.vectorizer.transform(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        features = self.transform(texts)
        probabilities = np.zeros((features.shape[0], len(self.models)), dtype=np.float32)
        for idx, model in enumerate(self.models):
            probabilities[:, idx] = model.predict_proba(features)[:, 1]
        return probabilities

