from sentence_transformers import SentenceTransformer
import json
import os
import numpy as np
import yaml

# Load config
with open(os.path.join(os.path.dirname(__file__), "../config/config.yaml")) as f:
    config = yaml.safe_load(f)

model = SentenceTransformer(config["embedding_model"])

def generate_embeddings(chunks_json_path, save_path):
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    embeddings = []
    for c in chunks:
        vec = model.encode(c["content"])
        embeddings.append({"chunk_id": c["chunk_id"], "vector": vec.tolist(), "metadata": c["metadata"]})
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, embeddings)
    print(f"Saved {len(embeddings)} embeddings to {save_path}")
    return embeddings

if __name__ == "__main__":
    generate_embeddings("../data/chunks/chunks.json", "../data/embeddings/chunks.npy")