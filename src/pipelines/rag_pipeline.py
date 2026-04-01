from generator.llm_client import LLMClient
from pipelines.context_builder import build_context, get_relevant_snippet
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
llm = LLMClient()

CHUNKS_FILE = "data/chunks/chunks.json"


def generate_answer(question, memory_context):
    chunks = build_context(CHUNKS_FILE, question, top_k=5)

    context_list = []
    context_text = ""

    for c in chunks:
        snippet = get_relevant_snippet(question, c["content"], model)
        context_list.append({
            "chunk_id": c["chunk_id"],
            "content": snippet,
            "metadata": c["metadata"]
        })
        context_text += snippet + "\n"

    prompt = f"""
Context:
{context_text}

Memory:
{memory_context}

Question:
{question}

Answer clearly:
"""

    answer = llm.generate(prompt)

    if len(answer) < 20:
        answer += " (refined: insufficient detail)"

    # 🔥 return BOTH
    return answer, context_list