#!/usr/bin/env python3
"""Reads a gensim Word2Vec model and returns its embeddings and features."""

import tensorflow as tf


def read_word2vec(filename):
    """Reads a gensim Word2Vec model.

    Args:
        filename: path to the saved Word2Vec model file

    Returns:
        embeddings: a numpy.ndarray of shape (vocab_size, vector_size)
        features: a list of the features (vocabulary words)
    """

    import gensim
    model = gensim.models.Word2Vec.load(filename)
    embeddings = model.wv.vectors
    features = model.wv.index_to_key

    return embeddings, features
