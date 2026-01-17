# Reddit Research Tool

A tool for downloading Reddit data from Arctic Shift API and performing RAG (Retrieval-Augmented Generation) queries over the collected data.

## What This Does

This project has two main components:

1. **Reddit Data Scraper** - Downloads posts and comments from Reddit subreddits using the Arctic Shift API
2. **RAG System** - Indexes the downloaded data using vector embeddings and allows you to ask questions about it

## Project Structure

```
.
├── Call_API.py                 # Advanced multi-worker scraper with partitioning
├── Get_reddit_data_class.py    # Simple class-based scraper
├── arctic_shift/               # Scraping utilities (API, partitioning, workers, etc.)
├── RAG/                        # RAG system components
│   ├── rag.py                  # Main RAG CLI
│   ├── pipeline.py             # Query pipeline
│   ├── embedder.py             # Embedding generation
│   ├── ingest.py               # Data ingestion
│   ├── retrievers.py           # Retrieval methods
│   ├── vector_store.py         # FAISS vector storage
│   ├── ai.py                   # LLM integration
│   └── api.py                  # FastAPI server
├── server/                     # Simple JSON receiver server
│   └── server.py
├── tests/                      # Test files
└── requirements.txt            # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Required for RAG embeddings
VOYAGE_AI_API_SECRET=your_voyage_api_key

# Optional: customize embedding model
VOYAGE_MODEL=voyage-2

# Optional: for chat-based answers
XAI_API_KEY=your_xai_api_key
XAI_BASE_URL=https://api.x.ai/v1
XAI_CHAT_MODEL=grok-4

# Optional: parallel embedding workers
RAG_EMBED_WORKERS=4
```

## Usage

### Downloading Reddit Data

#### Simple Scraper (Get_reddit_data_class.py)

Good for small datasets or testing:

```bash
# Download posts from r/programming for a specific day
python Get_reddit_data_class.py \
  -s programming \
  --after 2025-09-13 \
  --before 2025-09-14 \
  --what submissions \
  --outdir ./out

# Download both posts and comments
python Get_reddit_data_class.py \
  -s programming \
  --after 2025-09-13 \
  --before 2025-09-14 \
  --what both \
  --outdir ./out
```

#### Advanced Scraper (Call_API.py)

Better for large datasets with multi-worker partitioning:

```bash
# Download posts using 4 parallel workers
python Call_API.py \
  -s tressless \
  --after 2024-01-01 \
  --before 2025-01-01 \
  --workers 4 \
  --what posts \
  --outdir ./data

# Download both posts and comments
python Call_API.py \
  -s tressless \
  --workers 4 \
  --what both \
  --outdir ./data
```

**Options:**
- `-s, --subreddit`: Subreddit name (required)
- `--after`: Start date (ISO8601 format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
- `--before`: End date (ISO8601 format)
- `--workers`: Number of parallel workers (default: 4)
- `--what`: What to download: posts, comments, or both (default: posts)
- `--outdir`: Output directory
- `--force-histogram`: Force histogram-based partitioning (fail if unavailable)

### Building RAG Index

After downloading data, index it for semantic search:

```bash
# Index posts
python RAG/rag.py index-jsonl \
  --posts ./data/subreddit.posts.jsonl \
  --store .rag_store

# Index comments
python RAG/rag.py index-jsonl \
  --comments ./data/subreddit.comments.jsonl \
  --store .rag_store

# Index both at once
python RAG/rag.py index-jsonl \
  --posts ./data/subreddit.posts.jsonl \
  --comments ./data/subreddit.comments.jsonl \
  --store .rag_store
```

**Indexing Options:**
- `--index-mode`: Which indexes to build: embed (embeddings only), keyword (BM25 only), or both (default: both)
- `--embed-workers`: Number of parallel embedding workers (default: 1)
- `--faiss-write-y`: Write frequency for FAISS index (advanced)

### Querying the RAG System

Ask questions about your indexed data:

```bash
# Semantic search using embeddings
python RAG/rag.py ask \
  --query "What helps with hair loss?" \
  --k 8 \
  --store .rag_store \
  --mode embed

# Keyword search
python RAG/rag.py ask \
  --query "What helps with hair loss?" \
  --k 8 \
  --store .rag_store \
  --mode keyword

# Use both retrieval methods
python RAG/rag.py ask \
  --query "What helps with hair loss?" \
  --k 8 \
  --store .rag_store \
  --mode both

# Retrieve only (no answer generation)
python RAG/rag.py ask \
  --query "What helps with hair loss?" \
  --k 8 \
  --store .rag_store \
  --mode embed \
  --action retrieve
```

**Query Options:**
- `--query`: Your question (required)
- `--k`: Number of results to retrieve (default: 10)
- `--mode`: Retrieval method: embed, keyword, or both (default: embed)
- `--action`: retrieve (just get chunks) or ask (generate answer) (default: ask)
- `--store`: Path to RAG store directory (default: .rag_store)

### Running the RAG API Server

Start a FastAPI server for web integration:

```bash
uvicorn RAG.api:app --reload --port 8000
```

### Simple JSON Server

The server directory contains a basic HTTP server that receives JSON data (useful for browser extensions):

```bash
python server/server.py
```

This runs on port 8080 and accepts POST requests with JSON data.

## How It Works

### Scraping Pipeline

1. **Arctic Shift API** - Queries the Arctic Shift API for Reddit data
2. **Partitioning** - Splits large date ranges into partitions for parallel workers
3. **Histogram-based Distribution** - Uses aggregation histograms to balance work across workers
4. **Parallel Downloads** - Multiple workers download data simultaneously
5. **Merging** - Worker outputs are merged into final JSONL files

### RAG Pipeline

1. **Ingestion** - Reads posts/comments from JSONL files
2. **Embedding** - Generates vector embeddings using Voyage AI
3. **Indexing** - Stores embeddings in FAISS vector database
4. **Retrieval** - Finds relevant chunks using semantic search or keywords
5. **Generation** - Optionally generates answers using xAI's LLM

## Data Format

### Posts JSONL
Each line contains a post with fields: `id`, `created_utc`, `author`, `subreddit`, `score`, `title`, `selftext`, `url`, `num_comments`, etc.

### Comments JSONL
Each line contains a comment with fields: `id`, `created_utc`, `author`, `subreddit`, `score`, `body`, `link_id`, `parent_id`

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Tips

- For large subreddits, use more workers (8-16) to speed up downloads
- The advanced scraper (Call_API.py) automatically resumes interrupted downloads
- Build both embedding and keyword indexes for best retrieval results
- Use `--action retrieve` to see what chunks are being retrieved before generating answers
- Increase `--k` if you're not getting enough context for good answers

## Troubleshooting

**Rate limiting**: The Arctic Shift API has rate limits. The scraper will show rate limit headers.

**Embedding failures**: Make sure your VOYAGE_AI_API_SECRET is set correctly in .env

**Out of memory**: Reduce `--embed-workers` or process smaller batches

**No results**: Check that your date range contains data for the subreddit
