import faiss
import pickle
import numpy as np
from embeddings.clip_embedder import CLIPEmbedder

VECTOR_PATH = "data/cleaned/vector_store"

class ImageSearchEngine:
    def __init__(self):
        self.embedder = CLIPEmbedder()
        self.index = faiss.read_index(f"{VECTOR_PATH}/index.faiss")

        with open(f"{VECTOR_PATH}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def search_by_text(self, query, k=5):
        q_emb = self.embedder.embed_text(query)
        D, I = self.index.search(np.array([q_emb]), k)
        return D[0], I[0]

    def search_by_image(self, image_path, k=5):
        q_emb = self.embedder.embed_image(image_path)
        D, I = self.index.search(np.array([q_emb]), k)
        return D[0], I[0]

    def pretty_print(self, scores, indices):
        print("\nRetrieved Images:\n")

        for i, (score, idx) in enumerate(zip(scores, indices)):
            data = self.metadata[idx]

            print(f"Result {i+1}")
            print(f"Image: {data['image_path']}")
            print(f"Score: {score}")
            print(f"Caption: {data['caption']}")
            print(f"OCR Text: {data['ocr_text'][:300]}")
            print("-" * 50)

def main():
    engine = ImageSearchEngine()
    while(True):
        print("\nImage-RAG Query Engine\n")
        print("1 → Text to Image")
        print("2 → Image to Image")
        print("3 → Image to Text\n")

        mode = input("Select mode (1 / 2 / 3): ")

        if mode == "1":
            query = input("Enter text query: ")
            scores, indices = engine.search_by_text(query)

        elif mode == "2":
            path = input("Enter image path: ")
            scores, indices = engine.search_by_image(path)

        elif mode == "3":
            path = input("Enter image path: ")
            scores, indices = engine.search_by_image(path)

        else:
            print("Invalid mode")
            return

        engine.pretty_print(scores, indices)

if __name__ == "__main__":
    main()