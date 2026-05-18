"""Local embedding generation via fastembed (ONNX, no external service needed)."""

from __future__ import annotations

from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

# Module-level singleton — loaded once, reused for all calls.
_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(DEFAULT_MODEL)
    return _model


def ensure_model_available() -> bool:
    """Load (and if necessary download) the embedding model.

    Returns True on success, False if loading fails.
    First call downloads ~435 MB to ~/.cache/fastembed.
    """
    try:
        _get_model()
        return True
    except Exception:
        return False


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single text."""
    model = _get_model()
    return next(model.embed([text])).tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts (batched)."""
    if not texts:
        return []
    model = _get_model()
    return [emb.tolist() for emb in model.embed(texts)]
