import os
import json
from utils.file_loader import load_json  # assumes you have a function to load chunks.json
from retriever.hybrid_retriever import HybridRetriever
from retriever.reranker import Reranker
import re
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")
def get_relevant_snippet(query, text, model, top_n=2):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    if len(sentences) == 0:
        return text[:300]

    query_vec = model.encode(query, convert_to_tensor=True)
    sent_vecs = model.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(query_vec, sent_vecs)[0].cpu().numpy()
    top_idx = scores.argsort()[::-1][:top_n]

    selected = [sentences[i] for i in top_idx]

    return " ".join(selected)

def build_context(chunks_file, query, top_k=5):
    chunks = load_json(chunks_file)

    # Step 1: Retrieve using hybrid retriever
    retriever = HybridRetriever(chunks)
    retrieved = retriever.retrieve(query, top_k=top_k*3)  # fetch more for reranking

    # Step 2: Rerank top chunks
    reranker = Reranker()
    reranked = reranker.rerank(query, retrieved)

    # Step 3: Deduplicate and pick top_k
    seen_ids = set()
    final_context = []
    for c in reranked:
        if c["chunk_id"] not in seen_ids:
            final_context.append(c)
            seen_ids.add(c["chunk_id"])
        if len(final_context) >= top_k:
            break

    return final_context

if __name__ == "__main__":
    chunks_file = "data/chunks/chunks.json"
    query = input("Enter query: ").strip()
    context = build_context(chunks_file, query, top_k=5)

    print("\nFinal Context:\n")
    for c in context:
        snippet = get_relevant_snippet(query, c["content"], model)
        snippet = snippet[:300]   # limit length

        print(f"\nChunk ID: {c['chunk_id']}")
        print(f"Metadata: {c['metadata']}")
        print(f"Text: {snippet}")
        print("-" * 50)