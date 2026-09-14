"""
Retrieval layer for the RAG demo tab.

Two distinct paths, per the live-demo requirements:

1. Evaluated queries: the 5 queries that were manually judged in the
   notebook (pilot_retrieval_eval_df). Results for these come straight
   from that stored, labeled DataFrame -- never recomputed live -- so the
   displayed rank/score/relevant/P@5 values are exactly what was verified.

2. Custom query: a live FAISS search against the same 515 verified
   MiniLM embeddings, using the same model and index type the notebook
   used (all-MiniLM-L6-v2, IndexFlatIP over normalized vectors -- i.e.
   cosine similarity). These results carry NO relevance labels and NEVER
   feed back into the reported Mean Precision@5. `live_search` never
   raises: on any failure (no network, model download blocked, etc.) it
   returns an error string so the caller can fall back to the evaluated
   -query demonstration instead of crashing mid-demo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Building FAISS index from verified embeddings...")
def build_faiss_index(_embeddings: np.ndarray):
    """
    Rebuilds the FAISS index at app startup from the checkpointed 515
    embeddings. Leading underscore on the param tells Streamlit's cache
    not to hash the (large) array itself -- the index is cached for the
    life of the server process regardless.
    """
    import faiss

    dim = _embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(_embeddings)
    return index


@st.cache_resource(show_spinner="Loading MiniLM query encoder...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def live_search(
    query: str,
    index,
    chunks: list[dict],
    top_k: int = 5,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """
    Runs a live semantic search for an arbitrary custom query.

    Returns (results, error_message). Exactly one of the two is None.
    Never raises -- any failure (model download blocked, index issue,
    empty query, etc.) is returned as an error string so the UI can fall
    back gracefully instead of crashing the live demo.
    """
    query = (query or "").strip()
    if not query:
        return None, "Enter a query first."

    try:
        model = load_embedding_model()
        q_emb = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        scores, indices = index.search(q_emb, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(chunks):
                continue
            rec = chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "ticker": rec["ticker"],
                    "accession": rec["accession"],
                    "chunk_id": rec["chunk_id"],
                    "score": float(score),
                    "text": rec["text"],
                }
            )
        if not results:
            return None, "Search returned no results."
        return results, None

    except Exception as e:  # noqa: BLE001 - intentional catch-all for demo safety
        return None, f"Live retrieval unavailable ({type(e).__name__}: {e})"
