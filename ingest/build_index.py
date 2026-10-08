"""
Embeds every chunk and stores it in Chroma.

Have to be run again once the source papers change.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  

import chromadb
from sentence_transformers import SentenceTransformer

from ingest.extract import extract_all
from ingest.chunk import chunk_pages 

PAPERS_DIR = str(PROJECT_ROOT / "data" / "papers")
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def build_index():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nRunning ingestion pipeline")
    pages = extract_all(PAPERS_DIR)
    chunks = chunk_pages(pages)
    print(f"\n{len(pages)} pages into {len(chunks)} chunks to embed")

    texts = [c["text"] for c in chunks]
    ids = [f"{c['source']}-p{c['page']}-{c['chunk_index']}" for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

    print("\nEmbedding all chunks")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"\nWriting to Chroma at {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collections first 
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  

    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(), 
        documents=texts,
        metadatas=metadatas,
    )

    print(f"\n{collection.count()} chunks stored in Chroma")
    return model, collection


def test_retrieval(model, collection, question: str, n_results: int = 10):
    'Testing embedding of a real query.'

    print(f"\n--- Test query: \"{question}\" ---")
    query_embedding = model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
    )
    for i, (doc, meta, distance) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        print(f"\n[{i+1}] {meta['source']}, page {meta['page']} (distance: {distance:.3f})")
        print(doc[:300])


if __name__ == "__main__":
    model, collection = build_index()
    test_retrieval(model, collection, "What is the RMSE of MobileNet?")