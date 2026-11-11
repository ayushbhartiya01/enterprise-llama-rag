# Enterprise Llama RAG

A command-line question-answering application for asking focused questions about
PDFs. It parses PDFs with the LlamaParse v2 API, embeds chunks locally, stores
them in Qdrant, reranks retrieved passages with Cohere, and generates answers
with Groq.

> Open [`docs/index.html`](docs/index.html) for a visual overview of the
> components and data flow.

## What it does

- Ingests PDFs from `data/` with LlamaParse and splits extracted Markdown into
  chunks.
- Embeds chunks locally with `BAAI/bge-large-en-v1.5`.
- Stores and retrieves vectors in the Qdrant `enterprise_collection`.
- Reranks candidate passages with Cohere before checking their relevance.
- Answers focused questions with Groq's `openai/gpt-oss-120b` model.
- Rejects broad requests to summarize a document or the corpus. Ask a focused
  question or scope a summary to a section, page, or topic.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A running Qdrant instance
- API keys for [Groq](https://console.groq.com/keys), [Cohere](https://dashboard.cohere.com/api-keys),
  and [LlamaCloud](https://cloud.llamaindex.ai/)

The embedding model runs locally and is downloaded from Hugging Face on first
use. No Hugging Face token is required for public models, though rate limits
may apply.

## Setup

From the project directory:

```bash
# Create the virtual environment and install runtime dependencies.
uv sync --locked

# Copy the configuration template.
cp .env.example .env
```

Edit `.env` and set your API keys and Qdrant URL. Do not commit `.env` or
internal documents.

### Start local Qdrant

For local development, create a private Docker network and start Qdrant without
publishing its unauthenticated API port to the host:

```bash
docker network create enterprise-rag
docker run -d --name enterprise-rag-qdrant \
  --network enterprise-rag \
  --network-alias qdrant \
  -v "$PWD/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:v1.16.3
```

This unauthenticated Qdrant setup is for local development only. Use secured
Qdrant with authentication and TLS for shared or production deployments.

For local runs directly on your host, publish Qdrant only to loopback with
`-p 127.0.0.1:6333:6333` and use `QDRANT_URL=http://localhost:6333`. For a
managed or remote instance, use its URL instead.

## Add PDFs and ingest

Place PDFs in the project’s `data/` directory, then run:

```bash
uv run app.py --ingest
```

The command uploads PDFs to LlamaCloud for parsing, creates chunks and vectors,
and upserts them into Qdrant. The local `data/` directory is ignored by Git.

> **Data handling:** LlamaParse processes uploaded PDFs through LlamaCloud.
> Use only documents approved for that service. The current app does not
> implement authentication, authorization, or document-level RBAC; anyone who
> can access the running app can query the configured collection. Do not expose
> it to employees or production data until access controls are implemented and
> tested.

## Run with Docker

Build the image from the lockfile-pinned environment:

```bash
docker build -t enterprise-llama-rag:local .
```

The image runs as a non-root user. Provide credentials at runtime rather than
baking them into the image:

```bash
# Start the interactive app; Qdrant must be reachable from the container.
docker run --rm -it \
  --network enterprise-rag \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --pids-limit=256 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -v enterprise-rag-model-cache:/home/app/.cache/huggingface \
  --env-file .env \
  -e QDRANT_URL=http://qdrant:6333 \
  enterprise-llama-rag:local
```

For ingestion, mount the local documents read-only:

```bash
docker run --rm -it \
  --network enterprise-rag \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --pids-limit=256 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -v enterprise-rag-model-cache:/home/app/.cache/huggingface \
  --env-file .env \
  -e QDRANT_URL=http://qdrant:6333 \
  -v "$PWD/data:/app/data:ro" \
  enterprise-llama-rag:local --ingest
```

For a Qdrant instance on the Docker host, use
`QDRANT_URL=http://host.docker.internal:6333` instead. The image does not
include PDFs, credentials, or Qdrant data. The Hugging Face model cache is
stored in the container's writable home and is lost when the container is
removed unless you mount a persistent cache volume.

Linux installations, including the container, resolve CPU-only PyTorch wheels
to avoid installing CUDA libraries. Embedding inference therefore runs on CPU
on Linux. The image is multi-gigabyte because it includes PyTorch and the local
embedding stack; the Hugging Face model weights are downloaded on first use.

## Ask questions

After ingestion, start the interactive app:

```bash
uv run app.py
```

Example focused questions:

```text
What problem does the Transformer architecture address?
How does the paper describe positional encodings?
What does the policy say about employee data retention?
```

Enter `exit` or press Ctrl+C to quit. Broad prompts such as “summarize the
documents” are intentionally rejected. Use a focused question or specify a
section, page, or topic.

## Configuration

Copy `.env.example` to `.env`. The settings are read by Pydantic Settings.

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `GROQ_API_KEY` | Yes | — | Groq answer-generation API key |
| `COHERE_API_KEY` | Yes | — | Cohere reranking API key |
| `LLAMA_CLOUD_API_KEY` | Yes | — | LlamaParse v2 API key; currently required at app startup, and `LLAMAPARSE_API_KEY` is accepted for compatibility |
| `QDRANT_URL` | Yes | — | Qdrant endpoint, e.g. `http://localhost:6333` |
| `LLM_MODEL` | No | `openai/gpt-oss-120b` | Groq model ID |
| `EMBED_MODEL_NAME` | No | `BAAI/bge-large-en-v1.5` | Local Hugging Face embedding model |
| `SIMILARITY_TOP_K` | No | `20` | Number of Qdrant candidates retrieved |
| `RERANK_TOP_N` | No | `3` | Number of Cohere-ranked passages retained |
| `RERANK_MODEL` | No | `rerank-v5.0-pro` | Cohere reranking model |
| `GROUNDING_THRESHOLD` | No | `0.4` | Minimum top Cohere relevance score accepted by the grounding guard |

Threshold scores depend on the selected reranker and your documents. Tune the
threshold using representative queries and verify that the guard rejects
irrelevant passages; lowering it can allow weak context through.

## Tests

Install the optional test dependency and run the offline tests:

```bash
uv sync --locked --extra test
uv run pytest
```

The tests mock external parsing calls; they do not require API keys, a running
Qdrant service, or live model requests.

## Project layout

```text
.
├── app.py                         # CLI entry point and interactive query loop
├── data/                          # Local PDFs (ignored by Git)
├── docs/
│   └── index.html                 # Visual architecture and data-flow overview
├── src/llama_rag/
│   ├── config.py                  # Environment-backed settings
│   ├── database.py                # Qdrant client and collection
│   ├── guardrails.py              # Broad-summary and grounding checks
│   ├── parser.py                  # LlamaParse v2, Markdown chunking
│   └── pipeline.py                # Retrieval, reranking, and query engine
├── tests/
│   ├── conftest.py                # Isolated test configuration
│   ├── test_enterprise_guardrails.py
│   └── test_parser.py
├── .env.example                   # Configuration template (no real keys)
├── pyproject.toml                 # Project metadata and dependencies
└── uv.lock                        # Reproducible uv dependency lock
```
