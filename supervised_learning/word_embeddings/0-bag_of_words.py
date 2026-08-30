#!/usr/bin/env python3
"""Creates a bag of words embedding matrix."""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use, or None

    Returns:
        embeddings, features
    """
    # Clean and tokenize sentences
    tokenized = []

    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r"'s\b", "", sentence)
        words = re.findall(r"\b[a-zA-Z]+\b", sentence)
        tokenized.append(words)

    # Create vocabulary if not provided
    if vocab is None:
        features = sorted(set(
            word for sentence in tokenized for word in sentence
        ))
    else:
        features = list(vocab)

    # Create word -> column index mapping
    word_index = {word: i for i, word in enumerate(features)}

    # Create embedding matrix
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sentence in enumerate(tokenized):
        for word in sentence:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    return embeddings, np.array(features)
