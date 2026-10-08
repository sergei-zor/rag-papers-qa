"""
Separates extracted text into smaller overlapping chunks for embedding.
"""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 400

def chunk_text(text, chunk_size = CHUNK_SIZE, overlap = CHUNK_OVERLAP):
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def chunk_pages(pages):
    all_chunks = []
    for page in pages:
        for i, chunk in enumerate(chunk_text(page["text"])):
            all_chunks.append({
                "source": page["source"],
                "page": page["page"],
                "chunk_index": i,
                "text": chunk,
            })
        for t_i, table_text in enumerate(page.get("tables", [])):
            all_chunks.append({
                "source": page["source"],
                "page": page["page"],
                "chunk_index": f"table-{t_i}",
               "text": table_text,
            })
        for s_i, sentence in enumerate(page.get("table_sentences", [])):
            all_chunks.append({
                "source": page["source"],
                "page": page["page"],
                "chunk_index": f"table-sentence-{s_i}",
                "text": sentence,
            })
        for e_i, eq in enumerate(page.get("equations", [])):
            text = f"Equation ({eq['number']}): {eq['formula']}"
            if eq["context"]:
                text += f"\nText before: {eq['context']}"
            if eq.get("after"):
                text += f"\nText after: {eq['after']}"
            all_chunks.append({
                "source": page["source"],
                "page": page["page"],
                "chunk_index": f"equation-{e_i}",
                "text": text,
            })
    return all_chunks


if __name__ == "__main__":
    from ingest.extract import extract_all

    pages = extract_all("data/papers")
    chunks = chunk_pages(pages)
    print(f"\n{len(pages)} pages to {len(chunks)} chunks")

    table_chunks = [c for c in chunks if isinstance(c["chunk_index"], str)]
    print(f"{len(table_chunks)} of those are table chunks:\n")
    for c in table_chunks:
        print(f"--- {c['source']}, page {c['page']}, {c['chunk_index']} ---")
        print(c["text"])