import os
from PyPDF2 import PdfReader
import docx
import json

def load_text_files(folder_path):
    texts = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if fname.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                texts.append({"source": fname, "content": f.read()})
        elif fname.endswith(".pdf"):
            reader = PdfReader(fpath)
            for i, page in enumerate(reader.pages):
                texts.append({"source": fname, "page_number": i+1, "content": page.extract_text()})
        elif fname.endswith(".docx"):
            doc = docx.Document(fpath)
            full_text = "\n".join([p.text for p in doc.paragraphs])
            texts.append({"source": fname, "content": full_text})
    return texts


def load_json(path):
    """Load a JSON file and return as Python object."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)