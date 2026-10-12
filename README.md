# 🌐 Polyglot — Multimodal, Multilingual Chatbot

A production-oriented chat API and UI built on **FastAPI** and **Claude**. It
accepts text, images, voice notes, and uploaded documents; it decides what
language to answer in per turn; and it cites the documents it used.

```
Streamlit UI  ──HTTP/SSE──▶  FastAPI  ──▶  LLM provider layer  ──▶  Claude (vision + streaming)
                                │
                                ├─▶  faster-whisper   speech → text (auto-detects language)
                                ├─▶  edge-tts         text → speech (90+ locales)
                                ├─▶  RAG pipeline     extract → chunk → embed → retrieve
                                │                     pgvector (Postgres) | numpy (SQLite)
                                └─▶  PostgreSQL + Redis
```

---

## What's actually interesting here

Most chatbot projects are a wrapper around one API call. These are the decisions
that aren't:

**Language detection is a two-stage hybrid, not a library call.**
Statistical detectors are trained on paragraphs and fail on the 3–8 word
fragments people type into a chat box — a line of Devanagari confidently comes
back as English. So Unicode script is decided first (a hard fact, not a guess)
and narrows the answer to a family; the statistical detector then disambiguates
*within* that family, and only when there is enough text to justify overriding
the script's default. See [`app/services/language.py`](app/services/language.py).

**The system prompt is a frozen constant.**
Prompt caching is a byte-exact prefix match. Interpolating the reply language,
the user's name, or retrieved passages into the system prompt would invalidate
the cache on every turn — which is how most LLM apps end up paying full price
for a prompt they send thousands of times a day. Volatile per-request context is
delivered instead as a **mid-conversation `system` message**, which sits after
the cached prefix. See [`app/services/prompts.py`](app/services/prompts.py).

**Streaming opens its own database session.**
A `StreamingResponse` body is consumed *after* the endpoint returns, by which
point FastAPI has already closed the request-scoped session. Writing the
assistant message through it fails intermittently under load — the worst kind of
bug. `stream_turn` opens a short-lived session on completion, and there is a
regression test for it. See [`app/services/chat.py`](app/services/chat.py).

**Two vector-store backends behind one interface.**
`PgVectorStore` pushes ranking into PostgreSQL with pgvector's cosine operator
and an IVFFlat index. `NumpyVectorStore` ranks in-process so `git clone && pytest`
works with no extension and no Docker. Both write the same portable `float32`
BLOB, so switching is a config change plus a re-index.
See [`app/services/rag/store.py`](app/services/rag/store.py).

**Degrades instead of breaking.**
No Redis → in-process cache. No `sentence-transformers` → a lexical
character-n-gram embedder (and `/api/v1/ready` reports `semantic: false`, so a
degraded deployment can't masquerade as a healthy one). No `faster-whisper` →
speech-to-text returns a 503 naming the install command, and everything else
keeps working.

**Chunking is multilingual.**
The splitter's separator hierarchy includes `。` (CJK), `।` (Devanagari), and
`؟` (Arabic). A Latin-only splitter degrades to hard character cuts on half the
languages this app targets.

---

## Features

| | |
|---|---|
| 🖼 **Vision** | Images validated by decoding (not by `Content-Type`), EXIF stripped, downscaled to cap token cost |
| 🎤 **Speech in** | Whisper transcription with automatic language identification; the detected language drives the reply |
| 🔊 **Speech out** | Neural TTS in 90+ locales, no API key required |
| 📄 **Documents** | PDF / DOCX / TXT / MD / CSV → chunked, embedded, retrieved, cited inline as `[1]` |
| 🌍 **Cross-language retrieval** | A multilingual embedding space means a Hindi question can retrieve the English paragraph that answers it |
| ⚡ **Streaming** | Server-Sent Events with `start` / `delta` / `done` frames |
| 🔐 **Auth** | JWT access + refresh, bcrypt, per-user data isolation (404-not-403 on other users' resources) |
| 📊 **Cost accounting** | Per-message input/output/cache-read/cache-write tokens and latency, persisted |
| 🧮 **Token counting** | `POST /chat/estimate` returns the provider's *exact* count — never `tiktoken`, which is a different tokenizer and wrong for Claude |
| 🚦 **Rate limiting** | Fixed-window per user (per IP when anonymous), with `X-RateLimit-*` headers |
| 📝 **Observability** | Structured JSON logs with per-request correlation IDs propagated through `contextvars` |

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env            # macOS/Linux: cp .env.example .env
# set ANTHROPIC_API_KEY and JWT_SECRET_KEY in .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # for JWT_SECRET_KEY

alembic upgrade head
uvicorn app.main:app --reload      # API   → http://localhost:8000/docs
streamlit run frontend/streamlit_app.py   # UI → http://localhost:8501
```

**Without an API key**, set `LLM_PROVIDER=echo` in `.env`. The whole request path
— auth, rate limiting, persistence, streaming, RAG — runs against a deterministic
offline stub. This is what the test suite uses.

### Optional ML extras

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only, avoids a CUDA download
pip install -r requirements-ml.txt
```

This enables semantic (rather than lexical) retrieval and speech-to-text. Check
what's live at any time:

```bash
curl http://localhost:8000/api/v1/ready
```

### Full stack with Docker

```bash
docker compose up --build
```

Brings up PostgreSQL with pgvector, Redis, the API (migrations run first), and
the Streamlit UI. Requires `ANTHROPIC_API_KEY` and `JWT_SECRET_KEY` in `.env`.

---

## API

| Method | Path | |
|---|---|---|
| `POST` | `/api/v1/auth/register` · `/login` · `/refresh` | JWT issuing |
| `GET` `PATCH` | `/api/v1/auth/me` | Profile and language preference |
| `POST` | `/api/v1/chat` | Text + base64 images; `stream: true` for SSE |
| `POST` | `/api/v1/chat/multipart` | Binary image and voice-note uploads |
| `POST` | `/api/v1/chat/estimate` | Exact input-token count before sending |
| `GET` `PATCH` `DELETE` | `/api/v1/conversations[/{id}]` | History |
| `POST` | `/api/v1/media/transcribe` · `/speak` · `/detect-language` | Speech and language |
| `GET` | `/api/v1/media/voices` | Voice catalogue |
| `POST` `GET` `DELETE` | `/api/v1/documents[/{id}]` | Knowledge base |
| `POST` | `/api/v1/documents/search` | Retrieval without the model — isolates RAG bugs from generation bugs |
| `GET` | `/api/v1/health` · `/ready` | Liveness (dependency-free) and readiness |

Interactive docs at `/docs`; both docs and the OpenAPI schema are disabled when
`ENVIRONMENT=production`.

### Examples

```bash
TOKEN=$(curl -s localhost:8000/api/v1/auth/register \
  -H 'content-type: application/json' \
  -d '{"email":"a@b.com","password":"correct-horse-battery"}' | jq -r .access_token)

# Auto-detected language — asks in Hindi, answers in Hindi
curl localhost:8000/api/v1/chat -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"message":"मुझे छुट्टी की नीति समझाइए"}'

# Streaming
curl -N localhost:8000/api/v1/chat -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"message":"Explain RAG","stream":true}'

# Voice note + image in one turn
curl localhost:8000/api/v1/chat/multipart -H "Authorization: Bearer $TOKEN" \
  -F audio=@note.webm -F images=@chart.png -F message="What does this show?"

# Index a document, then ask about it
curl localhost:8000/api/v1/documents -H "Authorization: Bearer $TOKEN" -F file=@handbook.pdf
curl localhost:8000/api/v1/chat -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"message":"How many leave days do I get?"}'
```

---

## Layout

```
app/
├── main.py                  FastAPI app, middleware, lifespan
├── core/                    config · logging · security · errors · cache · rate limiting
├── db/                      async SQLAlchemy models and session
├── api/v1/                  route handlers
├── schemas/                 Pydantic request/response contracts
└── services/
    ├── llm/                 provider-agnostic interface + Claude adapter + offline stub
    ├── rag/                 extract · chunk · embed · store · pipeline
    ├── chat.py              turn orchestration
    ├── prompts.py           frozen system prompt + per-request directive
    ├── language.py          hybrid language identification
    ├── stt.py  tts.py       speech
    ├── images.py            vision preprocessing
    └── storage.py           upload storage
alembic/                     migrations (pgvector added on PostgreSQL only)
frontend/                    Streamlit UI + HTTP client
tests/                       93 tests, no network required
```

---

## Configuration

Every setting is in [`.env.example`](.env.example) with a working default.
The ones worth knowing:

| Variable | Note |
|---|---|
| `LLM_PROVIDER` | `anthropic` or `echo` (offline stub, no key needed) |
| `LLM_EFFORT` | `low` … `max`. `low` suits chat; raise for harder reasoning |
| `LLM_THINKING` | `adaptive` or `disabled`. Disabling is only valid at effort `high` or below — the app fails at **startup** with a message saying so, rather than 400ing per request |
| `DATABASE_URL` | SQLite by default; PostgreSQL switches the vector store to pgvector automatically |
| `REDIS_URL` | Empty falls back to an in-process cache — fine for one worker, not for many |
| `MAX_IMAGE_EDGE` | Long-edge pixel cap. Image tokens scale with area, so this is the largest cost lever in a vision app |
| `EMBEDDING_DIM` | Must match the embedding model. A mismatch is caught at model load, not after writing unreadable vectors |

---

## Testing

```bash
pytest -q          # 93 tests
```

Runs against the `echo` provider and a temporary SQLite file: no API key, no
network, no Docker. The suite covers language identification across ten scripts,
chunking (including CJK and pathological inputs), embedding round-trips,
per-user data isolation, SSE framing, and the streaming-persistence path.

---

## Known limitations

Stated plainly, because a project that claims to be production-grade should be
honest about its edges:

- **Document ingestion runs inline.** Bounded by a 20 MB upload ceiling, which
  keeps it in the low seconds. Above that it belongs on a task queue — the
  `DocumentStatus` states already model that transition.
- **`NumpyVectorStore` is O(n) per query** and capped at 20,000 candidate rows.
  Past a few tens of thousands of chunks, switch to PostgreSQL.
- **Scanned PDFs are rejected**, with an error naming OCR, rather than being
  silently indexed as empty.
- **The in-process cache and rate limiter are single-worker.** Set `REDIS_URL`
  before scaling past one process.
- **Refresh tokens are not revocable.** There is no denylist; a stolen refresh
  token is valid until it expires.
- **Storage is local disk.** `app/services/storage.py` is the only file that
  needs to change for S3.

## Possible next steps

CI (GitHub Actions running `ruff` + `pytest`), Prometheus metrics, Celery for
ingestion, a hybrid BM25 + dense retriever with reranking, and a token-budget
ceiling per user.
