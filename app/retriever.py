"""
Loads the embedding model and Chroma collection.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

model = None
collection = None

def load():
    global model, collection
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)
    if collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(COLLECTION_NAME)
    return model, collection

def retrieve(question: str, n_results: int = 6):
    """
    Embeds the question with the model used on chunks.
    """
    model, collection = load()
    query_embedding = model.encode([question])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

def initialize():
    """
    Forces the embedding model and Chroma to load immediately.
    """
    load()


def is_ready():
    return collection is not None


def chunk_count():
    return collection.count() if collection is not None else 0