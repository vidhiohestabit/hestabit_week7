import faiss
import numpy as np
import json

class Retriever:
    def __init__(self, index_path, chunks_path):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.vectors = np.array([c["content_vector"] for c in self.chunks], dtype=np.float32)

    def query(self, query_vec, top_k=5):
        D, I = self.index.search(np.array([query_vec]).astype(np.float32), top_k)
        results = [self.chunks[i] for i in I[0]]
        return results