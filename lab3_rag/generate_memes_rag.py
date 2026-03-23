import csv
import requests
import os

# Configuración
API_URL = os.getenv("API_URL", "http://localhost:5000")  # Cambia si tu backend Flask está en otro host/puerto
OUTPUT_CSV = "generated_memes_rag.csv"
TYPES = ["shaming", "stereotype", "objectification", "violence"]
EXAMPLES_PER_TYPE = 5
EXAMPLES_CONTEXT = 5  # Número de ejemplos reales a usar como contexto
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "training.csv"))

# Prompts base para cada tipo
PROMPTS = {
    "shaming": "Generate a new misogynistic meme text that uses shaming. The text should be in English and resemble real internet memes.",
    "stereotype": "Generate a new misogynistic meme text that uses stereotypes about women. The text should be in English and resemble real internet memes.",
    "objectification": "Generate a new misogynistic meme text that objectifies women. The text should be in English and resemble real internet memes.",
    "violence": "Generate a new misogynistic meme text that contains violent or threatening language towards women. The text should be in English and resemble real internet memes."
}

def load_examples_by_type(csv_path, types, n_examples=5):
    """Carga hasta n_examples reales por tipo desde el CSV."""
    groups = {t: [] for t in types}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for t in types:
                if row.get(t, "0") == "1":
                    text = row.get("Text Transcription") or row.get("text") or ""
                    if text.strip() and len(groups[t]) < n_examples:
                        groups[t].append(text.strip())
    return groups

def build_prompt_with_examples(base_prompt, examples):
    """Construye el prompt añadiendo ejemplos reales como contexto."""
    if not examples:
        return base_prompt
    context = "\n".join(f"Example: {ex}" for ex in examples)
    return f"{base_prompt}\n\nHere are some real meme examples for inspiration:\n{context}\n\nNow generate a new meme in the same style."

def generate_meme(prompt):
    resp = requests.post(
        f"{API_URL}/query",
        json={"question": prompt},
        timeout=120
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("answer", "")

def main():
    # Cargar ejemplos reales por tipo
    real_examples = load_examples_by_type(CSV_PATH, TYPES, n_examples=EXAMPLES_CONTEXT)
    rows = []
    for mtype in TYPES:
        prompt = build_prompt_with_examples(PROMPTS[mtype], real_examples[mtype])
        print(f"Generating {EXAMPLES_PER_TYPE} examples for type: {mtype}")
        for i in range(EXAMPLES_PER_TYPE):
            try:
                answer = generate_meme(prompt)
                print(f"  [{i+1}] {answer}")
                rows.append({"text": answer, "type": mtype})
            except Exception as e:
                print(f"  Error generating example {i+1} for {mtype}: {e}")
    # Guardar en CSV
    with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "type"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nGenerated {len(rows)} meme texts in {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
