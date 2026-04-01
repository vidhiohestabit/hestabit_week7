import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

class HybridRetriever:
    def __init__(self, chunks, embedding_model="all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.model = SentenceTransformer(embedding_model)
        self.chunk_texts = [c["content"] for c in chunks]
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.metadata = [c.get("metadata", {}) for c in chunks]

        # Precompute embeddings
        self.embeddings = self.model.encode(self.chunk_texts, convert_to_tensor=True)

        # Initialize BM25 for keyword fallback
        tokenized = [text.split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

    def mmr(self, query_vec, top_k=5, lambda_param=0.7):
        """
        Maximal Marginal Relevance
        """
        import torch
        embeddings = self.embeddings
        query_vec = torch.tensor(query_vec).unsqueeze(0)
        cosine_scores = util.cos_sim(query_vec, embeddings)[0].cpu().numpy()

        # Initialize
        selected = []
        candidate_idx = list(range(len(self.chunk_texts)))
        while len(selected) < top_k and candidate_idx:
            mmr_score = []
            for idx in candidate_idx:
                if selected:
                    similarity_to_selected = max([util.cos_sim(torch.tensor(self.embeddings[idx]), torch.tensor(self.embeddings[s])).item() for s in selected])
                else:
                    similarity_to_selected = 0
                score = lambda_param * cosine_scores[idx] - (1 - lambda_param) * similarity_to_selected
                mmr_score.append(score)
            best_idx = candidate_idx[np.argmax(mmr_score)]
            selected.append(best_idx)
            candidate_idx.remove(best_idx)
        return selected

    def retrieve(self, query, top_k=5, use_mmr=True):
        # Semantic search
        query_vec = self.model.encode(query, convert_to_tensor=True)

        # Keyword fallback scores
        tokenized_query = query.split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        bm25_scores = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores

        # Combined scores
        cosine_scores = util.cos_sim(query_vec, self.embeddings)[0].cpu().numpy()
        combined_scores = 0.7 * cosine_scores + 0.3 * bm25_scores

        if use_mmr:
            top_indices = self.mmr(query_vec, top_k=top_k)
        else:
            top_indices = combined_scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "metadata": self.metadata[idx],
                "content": self.chunk_texts[idx],
                "score": float(combined_scores[idx])
            })
        return results