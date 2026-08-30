#!/usr/bin/env python3
"""Creates and trains a Word2Vec model."""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds, and trains a Word2Vec model.

    Args:
        sentences: list of sentences to train on
        vector_size: dimensionality of the embedding layer
        min_count: minimum number of occurrences of a word
        window: maximum distance between current and predicted word
        negative: size of negative sampling
        cbow: True for CBOW, False for Skip-gram
        epochs: number of training iterations
        seed: random seed
        workers: number of worker threads

    Returns:
        The trained Word2Vec model
    """
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=0 if cbow else 1,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
