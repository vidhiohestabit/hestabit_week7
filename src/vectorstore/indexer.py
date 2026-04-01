import faiss
import numpy as np
import json
import os

def build_faiss_index(embeddings_path, index_path):
    embeddings_data = np.load(embeddings_path, allow_pickle=True)
    vectors = np.array([e["vector"] for e in embeddings_data]).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")
    return index

if __name__ == "__main__":
    build_faiss_index("../data/embeddings/chunks.npy", "../vectorstore/index.faiss")