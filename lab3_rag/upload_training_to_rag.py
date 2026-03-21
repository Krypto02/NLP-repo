import csv
import os
import requests
from io import BytesIO

# Configuración
API_URL = os.getenv("API_URL", "http://localhost:5000")
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "training.csv"))
TYPES = ["shaming", "stereotype", "objectification", "violence"]

# Lee el CSV y agrupa los textos por tipo de misoginia

def load_and_group_by_type(csv_path):
    groups = {t: [] for t in TYPES}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for t in TYPES:
                if row.get(t, "0") == "1":
                    text = row.get("Text Transcription") or row.get("text") or ""
                    if text.strip():
                        groups[t].append(text.strip())
    return groups

def upload_document(texts, doc_name):
    # Guarda los textos en un archivo temporal en memoria
    content = "\n\n".join(texts)
    file_bytes = content.encode("utf-8")
    files = {"file": (f"{doc_name}.txt", BytesIO(file_bytes))}
    resp = requests.post(f"{API_URL}/documents", files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()

def main():
    groups = load_and_group_by_type(CSV_PATH)
    for t, texts in groups.items():
        if not texts:
            print(f"No texts found for type: {t}")
            continue
        print(f"Uploading {len(texts)} texts for type: {t}")
        try:
            info = upload_document(texts, f"misogyny_{t}")
            print(f"  Uploaded as document: {info['document_id']} ({info['filename']})")
        except Exception as e:
            print(f"  Error uploading {t}: {e}")

if __name__ == "__main__":
    main()
