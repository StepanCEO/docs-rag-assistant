# docs-rag-assistant

RAG (Retrieval-Augmented Generation) assistant: index your own documents
(PDF / Markdown / TXT) and ask questions over them. Answers are grounded
strictly in the indexed content and always come with **citations** — if the
documents don't contain the answer, the assistant says so instead of guessing.

```
docs/ ──> loader ──> chunker (500 words, 80 overlap) ──> embeddings ──> ChromaDB
                                                                          │
question ──> embed ──> top-5 similar chunks ──> LLM (context-only) ──> answer + sources
```

Stack: plain OpenAI SDK, ChromaDB, FastAPI. No RAG frameworks — every step
(chunking, retrieval, grounding) is explicit and unit-tested.

## Quickstart

```bash
git clone https://github.com/StepanCEO/docs-rag-assistant
cd docs-rag-assistant
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY

# 1. Index the bundled sample docs (or your own folder)
python -m rag ingest sample_docs/

# 2. Ask
python -m rag ask "Сколько дней отпуска у сотрудников?"
```

```
$ python -m rag ask "Сколько дней отпуска у сотрудников?"

Сотрудники получают 28 оплачиваемых дней отпуска в год. Неиспользованные
дни не переносятся на следующий год.

Sources:
  - onboarding.md (chunk 0)
```

Questions can be in any language — the assistant answers in the language of
the question.

## API

```bash
uvicorn rag.api:app --reload
# or: docker compose up
```

- `POST /ask` — `{"question": "..."}` → `{"answer", "found", "sources"}`
- `GET /health`
- Interactive docs at `http://localhost:8000/docs`

## How grounding works

The LLM receives **only** the retrieved chunks as context and is instructed
to reply with a `NO_ANSWER` marker when the context is insufficient. The app
converts that marker into an honest "the documents don't cover this" response
with `found: false` — no hallucinated answers.

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | required |
| `CHAT_MODEL` | `gpt-4o-mini` | answering model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | embedding model |
| `CHUNK_SIZE` | `500` | words per chunk |
| `CHUNK_OVERLAP` | `80` | words shared between neighbours |
| `TOP_K` | `5` | chunks retrieved per question |

## Tests

```bash
pip install pytest
pytest
```

Tests run without an API key: a deterministic fake embedder exercises real
ChromaDB retrieval, and a fake LLM verifies grounding and refusal logic.

## Project structure

```
rag/
├── __main__.py         # CLI: ingest / ask
├── api.py              # FastAPI: POST /ask, GET /health
├── config.py           # .env settings
├── loader.py           # PDF/MD/TXT reading
├── chunker.py          # overlapping word-based chunking
├── store.py            # ChromaDB wrapper (pluggable embedder)
├── core.py             # retrieve -> grounded answer + citations
└── openai_backend.py   # embed/complete via OpenAI
tests/
├── test_chunker.py
└── test_rag.py         # end-to-end with fake embedder + fake LLM
sample_docs/            # demo documents for a 1-minute start
```

## License

MIT
