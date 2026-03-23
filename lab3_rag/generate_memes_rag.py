import csv
import requests
import os
import re
import random

# Configuración
API_URL = os.getenv("API_URL", "http://localhost:5000")
LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8080")
OUTPUT_CSV = "generated_memes_rag.csv"
MISOGYNISTIC_TYPES = ["shaming", "stereotype", "objectification", "violence"]
TOTAL_MEMES = 50
EXAMPLES_CONTEXT = 3
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "training.csv"))

# Prompts para memes misóginos
PROMPTS = {
    "shaming": (
        "Write one short internet meme caption in classic meme format (like 'Me: ... Also me: ...', "
        "'When you...', 'Nobody: ... Women: ...') that shames or belittles women for not meeting expectations. "
        "Be humorous and sarcastic in meme style. Output ONLY the meme text, one line, no explanation."
    ),
    "stereotype": (
        "Write one short internet meme caption in classic meme format (like 'Women be like...', "
        "'Every woman when...', 'POV: you ask a woman to...') that plays on stereotypes about women "
        "such as cooking, shopping, driving, or emotions. "
        "Be humorous in meme style. Output ONLY the meme text, one line, no explanation."
    ),
    "objectification": (
        "Write one short internet meme caption in classic meme format (like 'Women are just...', "
        "'The only thing women are good for...', 'POV: she thinks she\\'s a person') that reduces "
        "women to their physical appearance or domestic role. "
        "Output ONLY the meme text, one line, no explanation."
    ),
    "violence": (
        "Write one short internet meme caption in classic dark-humor meme format (like 'Me explaining to my wife...', "
        "'If she doesn\\'t obey...', 'When she says no:') that implies control or threats towards women. "
        "Output ONLY the meme text, one line, no explanation."
    ),
    "neutral": "Write a single short funny internet meme caption unrelated to gender. Output ONLY the meme text, one line, no explanation."
}

# Queries variadas para recuperar contexto RAG diverso por tipo
RAG_QUERIES = {
    "shaming": [
        "women failing responsibilities",
        "women not doing their share",
        "woman lazy irresponsible",
    ],
    "stereotype": [
        "women cooking driving shopping",
        "women bad at math logic",
        "women emotional irrational",
    ],
    "objectification": [
        "women physical appearance beauty",
        "women domestic role kitchen",
        "women as objects decoration",
    ],
    "violence": [
        "controlling women punish",
        "women obey or else",
        "threatening women force",
    ],
    "neutral": [
        "funny jokes everyday life",
    ],
}


def get_rag_context(mtype):
    """Recupera chunks del RAG con una query aleatoria del tipo dado."""
    query = random.choice(RAG_QUERIES.get(mtype, ["meme"]))
    try:
        resp = requests.post(
            f"{API_URL}/retrieve",
            json={"question": query, "top_k": 3},
            timeout=30
        )
        if resp.ok:
            chunks = resp.json().get("chunks", [])
            return " | ".join(c["text"][:120] for c in chunks if c.get("text"))
    except Exception:
        pass
    return ""


def load_examples_by_type(csv_path, types, n_examples=3):
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

def build_prompt(mtype, examples, already_generated=None):
    base = PROMPTS[mtype]
    parts = [base]
    if examples:
        exs = "\n".join(f"- {ex[:80]}" for ex in examples[:3])
        parts.append(f"Examples:\n{exs}")
    if already_generated:
        avoid = "\n".join(f"- {m[:80]}" for m in already_generated[-8:])
        parts.append(f"Do NOT repeat or closely paraphrase any of these already used memes:\n{avoid}")
    parts.append("Meme:")
    return "\n\n".join(parts)

def clean_meme(text):
    # Eliminar cualquier bloque entre corchetes o paréntesis
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    # Eliminar corchetes / paréntesis sueltos que queden
    text = re.sub(r'[\[\]()]+', '', text)
    # Eliminar URLs
    text = re.sub(r'http\S+', '', text)
    # Eliminar nombres de meme sites
    text = re.sub(r'\b(memecenter|meme\s*center|quickmeme|memegenerator|imgflip|roflbot|cheezburger)\b\S*', '', text, flags=re.IGNORECASE)
    # Eliminar hashtags y asteriscos
    text = re.sub(r'#\S*', '', text)
    text = re.sub(r'\*+', '', text)
    # Limpiar espacios múltiples
    text = re.sub(r'\s{2,}', ' ', text)

    # Recoger hasta 3 líneas válidas y unirlas con " / "
    good_lines = []
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'").rstrip(',.[]').strip()
        line = re.sub(r'^(meme:|example:|answer:|output:|\d+[\.\)])', '', line, flags=re.IGNORECASE).strip()
        if not line or line.upper() in ("OR", "-", "*"):
            continue
        if re.search(r'(the context|does not contain|not provide|provided context|cannot generate|no meme|no context|from sources|from context|unsuitable|threatening language)', line, re.IGNORECASE):
            continue
        if len(line) < 5:
            continue
        good_lines.append(line)
        if len(good_lines) >= 3:
            break

    if not good_lines:
        return ""

    result = " / ".join(good_lines)
    # Truncar si es demasiado largo
    if len(result) > 220:
        result = result[:220].rsplit(' ', 1)[0]
    return result

def generate_meme(prompt, context=""):
    # Construir prompt Mistral con contexto RAG opcional
    if context:
        full_prompt = f"[INST] Use this context as inspiration (do NOT copy it verbatim):\n{context}\n\n{prompt} [/INST]"
    else:
        full_prompt = f"[INST] {prompt} [/INST]"
    resp = requests.post(
        f"{LLAMA_URL}/completion",
        json={
            "prompt": full_prompt,
            "n_predict": 80,
            "temperature": 0.95,
            "top_p": 0.95,
            "repeat_penalty": 1.3,
            "stop": ["[INST]", "</s>", "\n\n"]
        },
        timeout=60
    )
    resp.raise_for_status()
    return resp.json().get("content", "")

def main():
    real_examples = load_examples_by_type(CSV_PATH, MISOGYNISTIC_TYPES, n_examples=EXAMPLES_CONTEXT)
    real_examples["neutral"] = []

    # 50 memes: ~60% misóginos, ~40% neutrales
    all_types = MISOGYNISTIC_TYPES * 3 + ["neutral"] * 2
    type_pool = (all_types * (TOTAL_MEMES // len(all_types) + 1))[:TOTAL_MEMES]
    random.shuffle(type_pool)

    rows = []
    used_memes = set()
    type_idx = 0

    while len(rows) < TOTAL_MEMES:
        mtype = type_pool[type_idx % len(type_pool)]
        type_idx += 1
        tries = 0
        meme_text = ""
        while tries < 5:
            already = [r["text"] for r in rows]
            prompt = build_prompt(mtype, real_examples.get(mtype, []), already_generated=already)
            try:
                context = get_rag_context(mtype)
                answer = generate_meme(prompt, context=context)
                candidate = clean_meme(answer)
                if candidate and candidate not in used_memes:
                    meme_text = candidate
                    used_memes.add(candidate)
                    break
            except Exception:
                pass
            tries += 1

        if meme_text:
            rows.append({"type": mtype if mtype != "neutral" else "meme", "text": meme_text})
            print(f"  [{len(rows)}/{TOTAL_MEMES}] {mtype}: {meme_text[:60]}")

    with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generados {len(rows)} memes en {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
