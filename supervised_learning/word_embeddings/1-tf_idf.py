#!/usr/bin/env python3
"""Creates a TF-IDF embedding matrix."""

import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use, or None

    Returns:
        embeddings, features
    """
    # Tokenize and clean sentences
    tokenized = []

    for sentence in sentences:
        sentence = sentence.lower()

        # Remove possessive "'s"
        sentence = re.sub(r"'s\b", "", sentence)

        # Extract words
        words = re.findall(r"\b[a-zA-Z]+\b", sentence)
        tokenized.append(words)

    # Create vocabulary if not provided
    if vocab is None:
        features = sorted(set(
            word for sentence in tokenized for word in sentence
        ))
    else:
        features = list(vocab)

    # Number of sentences
    s = len(sentences)

    # Number of features
    f = len(features)

    # Map each feature to its index
    word_index = {word: i for i, word in enumerate(features)}

    # Document frequency for each word
    df = np.zeros(f)

    for sentence in tokenized:
        unique_words = set(sentence)

        for word in unique_words:
            if word in word_index:
                df[word_index[word]] += 1

    # IDF
    idf = np.log((1 + s) / (1 + df)) + 1

    # TF-IDF embeddings
    embeddings = np.zeros((s, f))

    for i, sentence in enumerate(tokenized):
        for word in sentence:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

        # Multiply TF by IDF
        embeddings[i] *= idf

        # L2 normalization
        norm = np.linalg.norm(embeddings[i])

        if norm != 0:
            embeddings[i] /= norm

    return embeddings, np.array(features)
