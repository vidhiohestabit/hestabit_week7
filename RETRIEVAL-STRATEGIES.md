# Retrieval Strategies - Day 2

## 1. Hybrid Retrieval
- Combined FAISS semantic search + BM25 keyword fallback
- Ensures high recall: semantic search captures meaning, keywords capture exact matches

## 2. Reranking
- Cosine similarity of query embedding vs chunk embeddings
- Optionally, cross-encoder can be used for better precision

## 3. Deduplication
- Remove near-duplicate chunks based on first 200 characters
- Ensures final context is concise and non-repetitive

## 4. Context Window
- Top-K chunks (e.g., 5) selected based on reranking and deduplication
- Optimized to fit within LLM token limit

## 5. Traceable Context
- Each chunk keeps metadata: source file, page number, chunk_id
- Output shows full chunk content and metadata for transparency