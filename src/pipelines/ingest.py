import os
import json
import yaml
from math import ceil
from utils.file_loader import load_text_files
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ------------------------------
# Load config
# ------------------------------
with open(os.path.join(os.path.dirname(__file__), "../config/config.yaml")) as f:
    config = yaml.safe_load(f)

# ------------------------------
# Chunking function
# ------------------------------
def chunk_text(text, chunk_size=700, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = start + chunk_size
        chunk_content = " ".join(words[start:end])
        chunks.append({"chunk_id": chunk_id, "content": chunk_content})
        start += chunk_size - overlap
        chunk_id += 1
    return chunks

# ------------------------------
# Create chunks from raw files
# ------------------------------
def create_chunks(data_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    texts = load_text_files(data_folder)
    all_chunks = []

    global_chunk_id = 0  # global counter across all documents

    for doc in texts:
        doc_chunks = chunk_text(doc["content"], config["chunk_size"], config["chunk_overlap"])
        for c in doc_chunks:
            chunk_data = {
                "chunk_id": global_chunk_id,  # unique across all documents
                "content": c["content"],
                "metadata": {
                    "source": doc.get("source"),
                    "page_number": doc.get("page_number", None),
                    "tags": []
                }
            }
            all_chunks.append(chunk_data)
            global_chunk_id += 1  # increment for next chunk

    # Save chunks
    chunks_path = os.path.join(save_folder, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"Saved {len(all_chunks)} chunks to {chunks_path}")
    return chunks_path, all_chunks

# ------------------------------
# Generate embeddings for chunks
# ------------------------------
def generate_embeddings(chunks, model_name):
    model = SentenceTransformer(model_name)
    embeddings = []
    for c in chunks:
        vec = model.encode(c["content"])
        embeddings.append({"chunk_id": c["chunk_id"], "vector": vec.tolist(), "metadata": c["metadata"]})
    return embeddings

# ------------------------------
# Build FAISS index
# ------------------------------
def build_faiss_index(embeddings):
    vectors = np.array([e["vector"] for e in embeddings]).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

# ------------------------------
# User query function
# ------------------------------
# ------------------------------
# User query function
# ------------------------------
def query_loop(index, embeddings, model):
    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        # Encode query
        query_vec = model.encode(user_query).reshape(1, -1)

        # Search top 5 chunks
        D, I = index.search(query_vec.astype("float32"), 5)  # D = distances, I = indices

        print("\nResults:")
        for idx, dist in zip(I[0], D[0]):
            chunk = embeddings[idx]
            score = 1 / (1 + dist)  # optional: convert L2 distance to similarity-like score
            print(f"chunk_id: {chunk['chunk_id']}, source: {chunk['metadata']['source']}, score: {score:.6f}")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    # Get absolute paths based on the script location
    base_dir = os.path.dirname(__file__)  # src/pipelines
    data_folder = os.path.abspath(os.path.join(base_dir, "../data/raw/pdf"))
    chunks_folder = os.path.abspath(os.path.join(base_dir, "../data/chunks"))

    # Step 1: chunk the documents
    chunks_path, chunks = create_chunks(data_folder, chunks_folder)

    # Step 2: generate embeddings
    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(chunks, config["embedding_model"])

    # Step 3: build FAISS index
    print("\nBuilding FAISS index...")
    index = build_faiss_index(embeddings)
    print("Index built successfully!")

    # Step 4: start query loop
    model = SentenceTransformer(config["embedding_model"])
    query_loop(index, embeddings, model)