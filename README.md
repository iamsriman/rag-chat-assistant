# GenAI RAG Chat Assistant

A production-style Retrieval-Augmented Generation chatbot built with FastAPI, ChromaDB, Gemini embeddings, Groq LLM, and a responsive vanilla JavaScript frontend.

The assistant answers questions only from the knowledge base in `docs.json`. It performs real embedding-based retrieval before calling the LLM.

## Tech Stack

- Backend: Python, FastAPI
- Frontend: HTML, CSS, vanilla JavaScript
- Embeddings: Google Gemini `gemini-embedding-001`
- Vector database: ChromaDB persistent collection
- LLM: Groq OpenAI-compatible API
- Groq model: `llama-3.3-70b-versatile`
- Similarity: cosine similarity through ChromaDB cosine space

## Features

- Loads documents from `docs.json`
- Chunks document content before indexing
- Generates Gemini embeddings for every chunk
- Stores vectors and metadata in ChromaDB
- Retrieves top 3 chunks by vector similarity
- Applies a similarity threshold before generation
- Sends retrieved context to Groq
- Uses grounded prompt instructions
- Maintains last 5 message pairs by `sessionId`
- Returns source titles and similarity scores
- Handles empty requests, missing keys, timeouts, invalid keys, and rate limits
- Logs retrieved chunks, similarity scores, and token usage
- Provides a responsive chat UI with localStorage session handling

## Project Structure

```text
project/
├── app/
│   ├── routes/
│   ├── services/
│   ├── models/
│   ├── vectorstore/
│   ├── prompts/
│   ├── utils/
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── docs.json
├── requirements.txt
├── .env.example
└── README.md
```

## Environment Variables

Create `.env` from `.env.example`:

```env
GEMINI_API_KEY=your_real_gemini_api_key_here
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

GROQ_API_KEY=your_real_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BASE_URL=https://api.groq.com/openai/v1

RAG_TOP_K=3
RAG_SIMILARITY_THRESHOLD=0.25
REQUEST_TIMEOUT_SECONDS=20

CHROMA_DB_PATH=.chroma
CHROMA_COLLECTION=rag_documents
```

## Setup Instructions

Run these commands in PowerShell:

```powershell
cd C:\Users\P.Sriman\project-assignment
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your real Gemini and Groq API keys to `.env`.

Start the application:

```powershell
uvicorn app.main:app --reload --port 8001
```

Open the frontend:

```text
http://127.0.0.1:8001
```

Check backend health:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

Expected:

```json
{
  "status": "healthy"
}
```

## API Endpoints

### Health Check

`GET /health`

Response:

```json
{
  "status": "healthy"
}
```

### Chat

`POST /api/chat`

Request:

```json
{
  "sessionId": "abc123",
  "message": "How can I reset password?"
}
```

Response:

```json
{
  "reply": "Users can reset their password from Settings > Security > Password...",
  "tokensUsed": 120,
  "retrievedChunks": 3,
  "sources": [
    {
      "title": "Reset Password",
      "chunkId": "doc-0-chunk-0",
      "score": 0.86,
      "sourceDocument": "Reset Password"
    }
  ]
}
```

## Architecture

```text
docs.json
   |
   v
Load documents
   |
   v
Chunk documents
   |
   v
Gemini embeddings for chunks
   |
   v
Store vectors + metadata in ChromaDB
   |
   v
User question
   |
   v
Gemini query embedding
   |
   v
ChromaDB cosine similarity search
   |
   v
Top 3 chunks + similarity threshold
   |
   v
Grounded prompt with context + history + question
   |
   v
Groq llama-3.3-70b-versatile
   |
   v
Grounded response + sources + token usage
```

## RAG Workflow

### Indexing

1. Load all documents from `docs.json`.
2. Validate each document has `title` and `content`.
3. Split long content into overlapping chunks.
4. Generate Gemini embeddings for each chunk.
5. Store chunk text, embeddings, title, chunk id, and source document in ChromaDB.

### Querying

1. Embed the user question with Gemini using retrieval query mode.
2. Search ChromaDB using cosine similarity.
3. Retrieve the top `RAG_TOP_K` chunks.
4. Log similarity scores and retrieved chunk titles.
5. Filter chunks below `RAG_SIMILARITY_THRESHOLD`.
6. If no chunk passes the threshold, return:

```text
I could not find enough information in the knowledge base to answer this question.
```

7. Build a prompt using retrieved context, conversation history, and the user question.
8. Send the prompt to Groq.
9. Return the grounded answer with token usage and source metadata.

## Prompt Design

The prompt is designed to keep the model grounded:

```text
Use ONLY the provided context to answer.
Do not use outside knowledge.
If the answer is not present in the context, return the fallback response.
```

Conversation history is included for short-term continuity, but retrieved context remains the source of truth.

## Conversation Memory

The backend stores the last 5 user/assistant message pairs per `sessionId`.

The frontend stores the browser session id and displayed messages in localStorage, so refreshing the page keeps the chat visible.

## Validation and Error Handling

- Empty `message` or `sessionId` returns a structured `400` error.
- Missing Gemini key prevents indexing and returns degraded health.
- Missing Groq key returns a structured `503` error during chat.
- Provider timeouts return readable JSON errors.
- Invalid keys and rate limits are handled cleanly.
- Retrieval always happens before the LLM call.

## Example Questions

```text
How can I reset my password?
```

```text
Who can download invoices?
```

```text
What happens when API rate limits are exceeded?
```

```text
Who can export workspace data?
```

```text
What are the team roles?
```

## Screenshots

Add screenshots of the running app here:

```text
frontend chat screen
health endpoint response
sample RAG answer with sources
```

## Notes

ChromaDB persists vectors in `.chroma`, which is ignored by git. Delete `.chroma` if you want to force a completely fresh local vector database rebuild.
