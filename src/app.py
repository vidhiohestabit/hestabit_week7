import streamlit as st
from memory.memory_store import MemoryStore
from pipelines.rag_pipeline import generate_answer
from pipelines.sql_pipeline import run_sql_pipeline
from pipelines.image_pipeline import image_query_text, image_query_image
from evaluation.rag_eval import hallucination_score, confidence_score
from utils.logger import log

# init
if "memory" not in st.session_state:
    st.session_state.memory = MemoryStore()

memory = st.session_state.memory

st.title("🏢 Enterprise GenAI System")

tab1, tab2, tab3 = st.tabs(["💬 RAG", "🗄️ SQL", "🖼️ Image"])

# ================= RAG =================
with tab1:
    if "ans" not in st.session_state:
        st.session_state.ans = None
    if "context_used" not in st.session_state:
        st.session_state.context_used = None

    q = st.text_input("Ask question", key="rag")

    if st.button("Ask", key="rag_btn"):
        context_mem = memory.get()

        ans, context_used = generate_answer(q, context_mem)

        hall = hallucination_score(ans, context_used)
        conf = confidence_score(ans)

        memory.add(q, ans)

        log({
            "type": "rag",
            "q": q,
            "a": ans,
            "confidence": conf,
            "hallucination": hall
        })

        # 🔥 store in session
        st.session_state.ans = ans
        st.session_state.context_used = context_used
        st.session_state.conf = conf
        st.session_state.hall = hall

    # ✅ display safely
    if st.session_state.ans:
        st.write("### Answer")
        st.write(st.session_state.ans)

        st.write("### Scores")
        st.write("Confidence:", st.session_state.conf)
        st.write("Hallucination:", st.session_state.hall)

        st.write("### Top 5 Context")
        for i, c in enumerate(st.session_state.context_used):
            st.write(f"**Result {i+1}**")
            st.write(c["content"])
            st.divider()

# ================= SQL =================
with tab2:
    q = st.text_input("SQL question", key="sql")

    if st.button("Run SQL", key="sql_btn"):
        result = run_sql_pipeline(q)

        if "error" in result:
            st.error(result["error"])
        else:
            st.code(result["sql"])
            st.table(result["rows"])
            st.write(result["summary"])

# ================= IMAGE =================
with tab3:
    mode = st.radio("Mode", ["Text → Image", "Image → Image", "Image → Text"])

    # -------- TEXT → IMAGE --------
    if mode == "Text → Image":
        q = st.text_input("Enter query", key="img_text")

        if st.button("Search Image", key="btn_text_img"):
            results = image_query_text(q)

            for r in results:
                st.image(r["image"])
                st.write("Caption:", r["caption"])
                st.write("OCR:", r["ocr"][:200])
                st.write("Score:", r["score"])
                st.divider()

    # -------- IMAGE → IMAGE --------
    elif mode == "Image → Image":
        file = st.file_uploader("Upload Image", key="img_img")

        if file:
            path = f"temp_{file.name}"
            with open(path, "wb") as f:
                f.write(file.read())

            if st.button("Find Similar Images", key="btn_img_img"):
                results = image_query_image(path)

                for r in results:
                    st.image(r["image"])
                    st.write("Caption:", r["caption"])
                    st.write("Score:", r["score"])
                    st.divider()

    # -------- IMAGE → TEXT (NEW) --------
    elif mode == "Image → Text":
        file = st.file_uploader("Upload Image for OCR + Caption", key="img_text_mode")

        if file:
            st.image(file, caption="Uploaded Image")

            path = f"temp_{file.name}"
            with open(path, "wb") as f:
                f.write(file.read())

            if st.button("Extract Text & Caption", key="btn_img_text"):
                results = image_query_image(path)

                # take top result
                top = results[0]

                st.write("### Caption")
                st.write(top["caption"])

                st.write("### OCR Text")
                st.write(top["ocr"])

                st.write("### Score")
                st.write(top["score"])