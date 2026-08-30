#!/usr/bin/env python3
"""Builds and trains a gensim word2vec model"""
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                    negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds, and trains a gensim word2vec model"""
    sg = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers,
    )

    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs,
    )

    return model
