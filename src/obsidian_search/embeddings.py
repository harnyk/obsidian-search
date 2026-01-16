"""Ollama embedding generation."""

import ollama

DEFAULT_MODEL = "bge-m3"
EMBEDDING_DIM = 1024


def get_embedding(text: str, model: str = DEFAULT_MODEL) -> list[float]:
    """Generate embedding for a single text using Ollama."""
    response = ollama.embed(model=model, input=text)
    return response["embeddings"][0]


def get_embeddings_batch(
    texts: list[str],
    model: str = DEFAULT_MODEL
) -> list[list[float]]:
    """Generate embeddings for multiple texts using Ollama.

    Ollama's embed endpoint supports batching via the input parameter.
    """
    if not texts:
        return []

    response = ollama.embed(model=model, input=texts)
    return response["embeddings"]


def ensure_model_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if the embedding model is available, pull if not."""
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]

        # Check if model is available (may be listed with or without :latest tag)
        if model in model_names or f"{model}:latest" in model_names:
            return True

        # Try to pull the model
        ollama.pull(model)
        return True
    except Exception:
        return False
