# 🏢 Enterprise GenAI System — Deployment Notes

## 📌 Overview

This project is a **multimodal GenAI system** that supports:

* 📄 Text-based RAG (documents)
* 🖼️ Image RAG (CLIP + OCR + similarity search)
* 🗄️ SQL Question Answering (NL → SQL → Result)

The system is built using:

* Streamlit (UI)
* FAISS (vector store)
* Sentence Transformers (embeddings)
* SQLite (structured data)
* CLIP (image embeddings)
* Groq / LLM APIs (generation)

---

## 🧱 Project Structure

```
src/
│
├── app.py                     # Streamlit UI
├── pipelines/
│   ├── rag_pipeline.py
│   ├── sql_pipeline.py
│   ├── image_pipeline.py
│   └── context_builder.py
│
├── generator/
│   ├── llm_client.py
│   └── sql_generator.py
│
├── retriever/
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   └── image_search.py
│
├── embeddings/
│   ├── embedder.py
│   └── clip_embedder.py
│
├── memory/
│   └── memory_store.py
│
├── evaluation/
│   └── rag_eval.py
│
├── utils/
│   ├── schema_loader.py
│   ├── file_loader.py
│   └── logger.py
│
├── data/
│   ├── raw/
│   ├── chunks/
│   └── cleaned/
│
├── database/
│   └── sample.db
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd Week7/src
```

---

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Environment Variables

Create a `.env` file or export variables:

```bash
export GROQ_API_KEY=your_api_key_here
```

(Or use OpenAI / Gemini depending on your setup)

---

## 🗄️ Database Setup

Run:

```bash
python database/sample_db.py
```

This will create:

* `sample.db`
* `sales` table with sample data

---

## 📄 Document Ingestion (Day 1)

```bash
python pipelines/ingest.py
```

This will:

* Load documents
* Create chunks
* Generate embeddings
* Build FAISS index

---

## 🚀 Running the Application

```bash
streamlit run app.py
```

App will open at:

```
http://localhost:8501
```

---

## 🧠 Features

### 💬 RAG (Text)

* Hybrid retrieval (semantic + keyword)
* Reranking
* Memory (last 5 messages)
* Hallucination detection
* Confidence scoring

---

### 🗄️ SQL QA

* Natural language → SQL generation
* Schema-aware queries
* Safe execution on SQLite
* Result summarization

---

### 🖼️ Image RAG

Supports:

* Text → Image
* Image → Image
* Image → Text (OCR + retrieval)

Uses:

* CLIP embeddings
* FAISS index
* OCR text + captions

---

## 📊 Evaluation

Implemented:

* Hallucination Score
* Confidence Score
* Context traceability

---

## 🧠 Memory System

* Stores last 5 interactions
* Used for conversational continuity
* Stored in memory (session-based)

---

## 📝 Logging

Logs saved in:

```
CHAT-LOGS.json
```

Includes:

* Query
* Answer
* Confidence
* Hallucination score

---

## ⚠️ Important Notes

### ❌ Virtual Environment

Ensure `venv/` is in `.gitignore`:

```
venv/
```

---

### ❌ Temporary Files

Image uploads are processed **in-memory**
(No permanent storage)

---

### 🔐 API Keys

Never push:

* `.env`
* API keys

---

## 🚀 Deployment Options

### 1️⃣ Local (Recommended)

```bash
streamlit run app.py
```

---

### 2️⃣ Streamlit Cloud

* Push repo to GitHub
* Connect to Streamlit Cloud
* Add environment variables

---

### 3️⃣ Docker (Optional)

You can containerize using:

* Python base image
* Expose port 8501

---

## 🧪 Testing Checklist

✔ RAG returns answer + top 5 context
✔ SQL queries execute correctly
✔ Image search returns results
✔ Image upload does NOT store files
✔ Memory retains last 5 queries
✔ Logs are generated

---

## 🎯 Final Outcome

This system demonstrates:

* End-to-end RAG pipeline
* Multimodal retrieval (text + image)
* SQL reasoning
* Memory + evaluation
* Production-style architecture

---

## 👩‍💻 Author Notes

This project simulates an **enterprise knowledge intelligence system** capable of handling:

* Documents
* Images
* Structured databases

---

## 🚀 Future Improvements

* Redis-based memory
* Better reranking (cross-encoder)
* Real-time streaming responses
* UI enhancements
* Cloud deployment (AWS/GCP)

---
