def hallucination_score(answer, context):
    # 🔥 handle list case
    if isinstance(context, list):
        context = " ".join([c["content"] for c in context])

    words = answer.lower().split()

    matches = sum(1 for w in words if w in context.lower())

    return round(matches / len(words), 2) if words else 0
def confidence_score(answer):
    return round(min(len(answer) / 150, 1), 2)