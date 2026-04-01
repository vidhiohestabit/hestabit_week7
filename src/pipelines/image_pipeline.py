from retriever.image_search import ImageSearchEngine

engine = ImageSearchEngine()

def image_query_text(query):
    scores, indices = engine.search_by_text(query)
    results = []

    for score, idx in zip(scores, indices):
        data = engine.metadata[idx]
        results.append({
            "image": data["image_path"],
            "caption": data["caption"],
            "ocr": data["ocr_text"],
            "score": float(score)
        })

    return results


def image_query_image(image_path):
    scores, indices = engine.search_by_image(image_path)
    results = []

    for score, idx in zip(scores, indices):
        data = engine.metadata[idx]
        results.append({
            "image": data["image_path"],
            "caption": data["caption"],
            "ocr": data["ocr_text"],
            "score": float(score)
        })

    return results