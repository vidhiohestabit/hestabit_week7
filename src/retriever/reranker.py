from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks):
        # chunks: list of dicts with 'content'
        texts = [c["content"] for c in chunks]
        scores = self.model.predict([[query, t] for t in texts])
        for i, c in enumerate(chunks):
            c["score"] = float(scores[i])
        # Sort by score descending
        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks