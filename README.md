# rag-papers-qa
![CI](https://github.com/sergei-zor/rag-papers-qa/actions/workflows/ci.yml/badge.svg)

## Question answering over scientific papers on nanoporous metals

This repository contains a retrieval-augmented generation (RAG) service that answers questions about a small collection of papers on nanoporous metals (molecular dynamics, genetic algorithms and 3D CNNs), with the source file and page for every retrieved passage. The answers are generated either by a local [Ollama](https://ollama.com) model or by Amazon Bedrock. 

```
PDFs --> extract --> chunk --> embed --> ChromaDB
                                            |
question --> embed --> top-k chunks --------+
                           |
                  prompt --> LLM (Ollama | Bedrock) --> answer + sources

```

### Repository layout

```
ingest/         extract.py, chunk.py, build_index.py
app/            main.py (FastAPI), retriever.py, generator.py, llm_ollama.py, llm_bedrock.py
data/papers/    the PDFs
Dockerfile, docker-compose.yml, docker-compose.ollama.yml
```

### Papers

The `data/papers/` folder contains the PDFs the RAG is built on (co-authored by the owner of this repository):

| File | Publication |
|---|---|
| A_molecular_dynamics_study.pdf | [J. Appl. Phys. 138, 075103 (2025)](https://doi.org/10.1063/5.0278347) |
| Transferable_3D_CNNs.pdf | [Mater. Des., vol. 260, p. 114896, 2025.](https://doi.org/10.1016/j.matdes.2025.114896) |
| SI_Transferable_3D_CNNs.pdf | its supplementary information |
| Does_high_entropy_improves.pdf | [Comput. Mater. Sci. 262, 114332 (2026)](https://doi.org/10.1016/j.commatsci.2025.114332) (not included, subscription article still under embargo) |
| Bayesian_optimization_of_genetic.pdf | [arXiv preprint, arXiv:2607.07289, 2026.](https://doi.org/10.48550/arXiv.2607.07289) |
| tda_segmentor_a_tool.pdf | [arXiv preprint, arXiv:2312.16558, 2024.](https://doi.org/10.48550/arXiv.2312.16558) |

To use the service on other papers, replace the files in `data/papers/` and rebuild the index.


### Index preparation

To build the index, run:

```
python ingest/build_index.py
```

This extracts the text, tables and equations from every PDF in `data/papers/`, splits them into chunks, embed them with `all-MiniLM-L6-v2` and writes a ChromaDB collection to `chroma_db/`. The embedding model is downloaded from the HuggingFace hub. 
There is no automatic update of the index. After addition or change in `data/papers/`, run the index building script again.

### Running the service

Install [Ollama](https://ollama.com), pull a model and start the API:

```
ollama pull qwen3.5:4b
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Then ask a question:

```
curl -s localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "What R2 did DenseNet-201 achieve for elastic constant prediction?", "n_results": 6}'
```

The response contains the generated answer and the source file and page of each retrieved chunk. `GET /health` returns the service status.

The LLM backend is chosen with environment variables:

```
LLM_PROVIDER=ollama | bedrock          default: ollama
OLLAMA_MODEL=qwen3.5:4b                default: qwen3.5:4b
OLLAMA_HOST=http://localhost:11434     default: http://localhost:11434
BEDROCK_MODEL_ID=eu.amazon.nova-micro-v1:0  default: eu.amazon.nova-micro-v1:0
AWS_REGION=eu-north-1                  default: eu-north-1
```

Questions are limited to 1000 characters and `n_results` to 10. If the LLM backend is unreachable or returns an error, `/query` responds with HTTP 502.

On Windows PowerShell, set a variable with `$env:LLM_PROVIDER = "bedrock"`.

### Running in Docker

The image contains the embedding model and a prebuilt index, so a container starts instantly and does not need the network for retrieval. The PDFs are part of the repository, so the image builds from a .

To use a local Ollama server running:

```
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up --build
```

To use Amazon Bedrock (the default):

```
docker compose up --build
```

After changing the PDFs, build the image again.

### Credits
Embeddings: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Vector store: [ChromaDB](https://www.trychroma.com). PDF processing: [pdfplumber](https://github.com/jsvine/pdfplumber).

The 3D CNN model discussed in the papers is a part of [cnn-nanoporous](https://github.com/sergei-zor/cnn-nanoporous).

### License
The code is released under the MIT License (see `LICENSE`). The papers in `data/papers/` are not covered by it and keep their licenses from publishers.