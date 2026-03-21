import csv
import requests
import os

# Configuración
API_URL = os.getenv("API_URL", "http://localhost:5000")  # Cambia si tu backend Flask está en otro host/puerto
OUTPUT_CSV = "generated_memes_rag.csv"
TYPES = ["shaming", "stereotype", "objectification", "violence"]
EXAMPLES_PER_TYPE = 5

# Prompts base para cada tipo
PROMPTS = {
    "shaming": "Generate a new misogynistic meme text that uses shaming. The text should be in English and resemble real internet memes.",
    "stereotype": "Generate a new misogynistic meme text that uses stereotypes about women. The text should be in English and resemble real internet memes.",
    "objectification": "Generate a new misogynistic meme text that objectifies women. The text should be in English and resemble real internet memes.",
    "violence": "Generate a new misogynistic meme text that contains violent or threatening language towards women. The text should be in English and resemble real internet memes."
}

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
    rows = []
    for mtype in TYPES:
        prompt = PROMPTS[mtype]
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
