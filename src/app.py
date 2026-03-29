"""
Misogyny Classifier — Live Demo  (v6 · neon)
Run with:  streamlit run src/app.py
"""

import csv
import json
import math
import os
import random
import sys
import time

import requests
import streamlit as st
import torch
from transformers import AutoTokenizer

# Make the backend scripts importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "scripts"))
from generate_memes_rag import (  # noqa: E402  # pylint: disable=wrong-import-position,wrong-import-order
    build_prompt,
    clean_meme,
    generate_meme,
    get_rag_context,
    load_examples_by_type,
    MISOGYNISTIC_TYPES,
)

from model import MultiTaskModel  # noqa: E402  # pylint: disable=wrong-import-position

# ── Config ────────────────────────────────────────────────────────────────────
BERT_MODEL = "cardiffnlp/twitter-roberta-base-hate"
BERT_MAX_LEN = 128
LABEL_COLS = ["shaming", "stereotype", "objectification", "violence"]
LABEL_META = {
    "shaming": {"icon": "😳", "grad": ["#ff6b6b", "#ee5a24"], "color": "#ff6b6b"},
    "stereotype": {"icon": "🏷️", "grad": ["#ffa502", "#e67e22"], "color": "#ffa502"},
    "objectification": {"icon": "👁️", "grad": ["#a55eea", "#8854d0"], "color": "#a55eea"},
    "violence": {"icon": "⚡", "grad": ["#ff4757", "#c0392b"], "color": "#ff4757"},
}
LABEL_DESC = {
    "shaming": "Body / slut shaming",
    "stereotype": "Gender stereotypes",
    "objectification": "Sexual objectification",
    "violence": "Threats / physical violence",
}
_HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(_HERE, "models", "trained")
CSV_PATH = os.path.join(_HERE, "data", "training", "training.csv")

# Generator endpoints (Docker Compose services)
LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8080")
RAG_API = os.getenv("API_URL", "http://localhost:5000")

EXAMPLES_SAFE = [
    "Javier we love you, let us pass the exam :)",
]
EXAMPLES_TOXIC = [
    "She is too ugly to be loved by anyone.",
    "The only thing women are good for is cooking and cleaning.",
    "She got promoted because of her looks, not her brain.",
    "Shut up woman, go make me a sandwich.",
]


@st.cache_data(show_spinner=False)
def load_dataset_texts():
    """Load meme captions from training.csv for the random sampler."""
    texts = {"safe": [], "misogynous": []}
    if not os.path.isfile(CSV_PATH):
        return texts
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            caption = (row.get("Text Transcription") or "").strip()
            if not caption or len(caption) < 10 or len(caption) > 280:
                continue
            if row.get("misogynous") == "1":
                texts["misogynous"].append(caption)
            else:
                texts["safe"].append(caption)
    return texts


@st.cache_data(show_spinner=False)
def _load_gen_examples():
    """Load few-shot examples from training.csv for the generator."""
    return load_examples_by_type(CSV_PATH, MISOGYNISTIC_TYPES, n_examples=3)


def generate_meme_live(category):
    """Call llama.cpp via generate_memes_rag functions. Returns (text, used_rag)."""
    examples = _load_gen_examples().get(category, [])
    prompt = build_prompt(category, examples)
    rag_ctx = get_rag_context(category)
    raw = generate_meme(prompt, context=rag_ctx)
    return clean_meme(raw), bool(rag_ctx)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Misogyny Classifier", page_icon="🛡️", layout="centered")

# ── CSS (injected via st.html so it never leaks as text) ─────────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
:root {
    --bg:        #060910;
    --surface:   #0d1525;
    --surface2:  #141e35;
    --border:    #243056;
    --text:      #e2e8f0;
    --text2:     #8492b4;
    --accent:    #a78bfa;
    --cyan:      #22d3ee;
    --green:     #34d399;
    --red:       #fb7185;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], .main, .block-container,
[data-testid="stAppViewContainer"] *, .block-container * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    background-image:
        linear-gradient(rgba(167,139,250,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(167,139,250,.03) 1px, transparent 1px),
        radial-gradient(ellipse 900px 600px at 15% 10%, rgba(167,139,250,.14), transparent),
        radial-gradient(ellipse 700px 500px at 85% 30%, rgba(34,211,238,.10), transparent),
        radial-gradient(ellipse 600px 700px at 50% 95%, rgba(52,211,153,.07), transparent) !important;
    background-size: 80px 80px, 80px 80px, 100%, 100%, 100% !important;
    background-attachment: fixed !important;
}
.main { background: transparent !important; }
.block-container { max-width: 900px !important; padding: 2rem 1.6rem 5rem !important; }
[data-testid="stSidebar"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
footer, [data-testid="stDecoration"], #MainMenu { display: none !important; }

/* ── Hero ─────────────────────────────────────── */
.hero {
    position: relative; overflow: hidden;
    background:
        linear-gradient(var(--surface), var(--surface)) padding-box,
        linear-gradient(90deg, #a78bfa, #6366f1, #22d3ee, #34d399, #f97316, #a78bfa) border-box;
    border: 2.5px solid transparent;
    border-radius: 28px;
    padding: 4.5rem 2.4rem 3.5rem;
    text-align: center;
    margin-bottom: 2rem;
    color: var(--text);
    background-size: 100% 100%, 300% 100%;
    animation: heroBorder 6s linear infinite;
    box-shadow:
        0 4px 100px rgba(167,139,250,.18),
        0 0 200px rgba(34,211,238,.08),
        0 0 0 1px rgba(167,139,250,.1),
        inset 0 1px 0 rgba(255,255,255,.08),
        inset 0 -1px 0 rgba(0,0,0,.3);
}

/* dot grid overlay */
.hero-dots {
    position: absolute; inset: 0; z-index: 0; pointer-events: none; opacity: .45;
    background-image: radial-gradient(circle, rgba(167,139,250,.35) 1px, transparent 1px);
    background-size: 24px 24px;
    mask-image: radial-gradient(ellipse 80% 80% at 50% 40%, black 20%, transparent 70%);
    -webkit-mask-image: radial-gradient(ellipse 80% 80% at 50% 40%, black 20%, transparent 70%);
}

/* floating orbs */
.hero-bg {
    position: absolute; inset: 0; z-index: 0; overflow: hidden; pointer-events: none;
}
.hero-bg .orb { position: absolute; border-radius: 50%; }
.hero-bg .orb1 {
    width: 420px; height: 420px; top: -150px; right: -80px;
    background: radial-gradient(circle, rgba(167,139,250,.32), rgba(167,139,250,.02) 70%);
    animation: orbDrift1 14s ease-in-out infinite alternate;
}
.hero-bg .orb2 {
    width: 360px; height: 360px; bottom: -130px; left: -70px;
    background: radial-gradient(circle, rgba(34,211,238,.28), rgba(34,211,238,.01) 70%);
    animation: orbDrift2 18s ease-in-out infinite alternate;
}
.hero-bg .orb3 {
    width: 260px; height: 260px; top: 30%; left: 55%;
    background: radial-gradient(circle, rgba(52,211,153,.22), transparent 70%);
    animation: orbDrift3 11s ease-in-out infinite alternate;
}

/* scan line */
.hero-scan {
    position: absolute; left: 0; right: 0; height: 2px; z-index: 1;
    pointer-events: none;
    background: linear-gradient(90deg, transparent, rgba(167,139,250,.45), transparent);
    box-shadow: 0 0 20px rgba(167,139,250,.35);
    animation: scanMove 5s ease-in-out infinite;
}

/* badge */
.hero-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 92px; height: 92px; border-radius: 26px;
    background:
        linear-gradient(rgba(167,139,250,.12), rgba(34,211,238,.08)) padding-box,
        linear-gradient(135deg, rgba(167,139,250,.65), rgba(34,211,238,.45)) border-box;
    border: 2px solid transparent;
    font-size: 2.6rem; margin-bottom: 1.5rem;
    box-shadow:
        0 8px 50px rgba(167,139,250,.3),
        0 0 100px rgba(167,139,250,.12),
        0 0 0 8px rgba(167,139,250,.06),
        inset 0 1px 0 rgba(255,255,255,.1);
    animation: badgeFloat 3.5s ease-in-out infinite, badgeGlow 3s ease-in-out infinite;
    position: relative; z-index: 2;
    backdrop-filter: blur(12px);
}

/* title */
.hero h1 {
    font-size: 3.5rem; font-weight: 900; margin: 0;
    letter-spacing: -3px; line-height: 1;
    color: #fff; position: relative; z-index: 2;
    text-shadow: 0 0 60px rgba(167,139,250,.15);
}
.hero h1 .grad {
    background: linear-gradient(135deg, #a78bfa 0%, #22d3ee 40%, #34d399 80%, #a78bfa 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: gradShift 6s ease-in-out infinite alternate;
    filter: drop-shadow(0 0 30px rgba(167,139,250,.3));
}
.hero .tagline {
    font-size: .88rem; color: var(--text2); margin-top: .8rem;
    font-weight: 400; line-height: 1.6; letter-spacing: .3px;
    position: relative; z-index: 2;
}

/* stats bar */
.hero-stats {
    display: flex; justify-content: center; gap: 2.5rem;
    margin-top: 2rem; position: relative; z-index: 2;
}
.hero-stat { display: flex; flex-direction: column; align-items: center; }
.hero-stat .stat-val {
    font-size: 1.65rem; font-weight: 900; letter-spacing: -1px;
    background: linear-gradient(135deg, #a78bfa, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    filter: drop-shadow(0 0 12px rgba(167,139,250,.4));
}
.hero-stat .stat-lbl {
    font-size: .62rem; font-weight: 600; color: var(--text2);
    text-transform: uppercase; letter-spacing: 1px; margin-top: 2px;
}

/* pills */
.hero .pills {
    display: flex; justify-content: center; gap: 8px;
    margin-top: 1.6rem; flex-wrap: wrap;
    position: relative; z-index: 2;
}
.hero .pill {
    background:
        linear-gradient(rgba(167,139,250,.06), rgba(167,139,250,.02)) padding-box,
        linear-gradient(135deg, rgba(167,139,250,.35), rgba(34,211,238,.22)) border-box;
    border: 1px solid transparent;
    border-radius: 999px; padding: 6px 16px;
    font-size: .68rem; font-weight: 600; color: #c4b5fd;
    backdrop-filter: blur(6px);
    transition: all .3s ease;
}
.hero .pill:hover {
    background:
        linear-gradient(rgba(167,139,250,.15), rgba(167,139,250,.08)) padding-box,
        linear-gradient(135deg, rgba(167,139,250,.55), rgba(34,211,238,.35)) border-box;
    box-shadow: 0 0 30px rgba(167,139,250,.2);
    transform: translateY(-2px);
}

/* ── Section label ────────────────────────────── */
.sec-label {
    font-size: .68rem; font-weight: 700; color: var(--text2);
    text-transform: uppercase; letter-spacing: 2px;
    margin-bottom: .6rem; margin-top: .2rem;
    display: flex; align-items: center; gap: 10px;
}
.sec-label::before {
    content: ''; width: 3px; height: 18px; border-radius: 3px;
    background: linear-gradient(180deg, #a78bfa, #22d3ee);
    flex-shrink: 0;
    box-shadow: 0 0 12px rgba(167,139,250,.4);
}

/* ── Streamlit text_area ──────────────────────── */
[data-testid="stTextArea"] textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 16px !important;
    color: var(--text) !important;
    font-size: .92rem !important;
    padding: 16px 18px !important;
    transition: border-color .25s, box-shadow .25s;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,.12), 0 0 40px rgba(167,139,250,.08) !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: var(--text2) !important; }
[data-testid="stTextArea"] label { display: none !important; }

/* ── Primary button ───────────────────────────── */
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #a78bfa, #7c3aed, #a78bfa, #22d3ee) !important;
    background-size: 300% 100% !important;
    animation: shimmer 4s ease infinite !important;
    border: none !important;
    border-radius: 18px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 18px 0 !important;
    letter-spacing: .5px;
    color: #fff !important;
    box-shadow:
        0 4px 30px rgba(167,139,250,.4),
        0 0 80px rgba(167,139,250,.15),
        inset 0 1px 0 rgba(255,255,255,.15) !important;
    transition: transform .15s, box-shadow .2s;
    text-shadow: 0 1px 2px rgba(0,0,0,.3);
}
[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow:
        0 8px 50px rgba(167,139,250,.5),
        0 0 120px rgba(167,139,250,.2) !important;
}

/* ── Secondary buttons ────────────────────────── */
[data-testid="stButton"] > button[kind="secondary"],
[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
    padding: 10px 8px !important;
    transition: all .25s;
}
[data-testid="stButton"] > button[kind="secondary"]:hover,
[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover {
    border-color: rgba(167,139,250,.4) !important;
    background: rgba(167,139,250,.08) !important;
    box-shadow: 0 0 25px rgba(167,139,250,.08) !important;
    transform: translateY(-1px);
}

/* ── Result panel ─────────────────────────────── */
.result-panel {
    position: relative; overflow: hidden;
    background: var(--surface);
    border-radius: 24px;
    padding: 2.5rem 2rem 2rem;
    margin: 1.4rem 0;
    animation: slideUp .55s cubic-bezier(.22,1,.36,1);
}
.result-panel.safe {
    background:
        linear-gradient(var(--surface), var(--surface)) padding-box,
        linear-gradient(135deg, rgba(52,211,153,.6), rgba(34,211,238,.5), rgba(52,211,153,.4)) border-box;
    border: 2px solid transparent;
    box-shadow:
        0 0 60px rgba(52,211,153,.12),
        0 0 120px rgba(52,211,153,.05),
        0 0 0 1px rgba(52,211,153,.08),
        inset 0 1px 0 rgba(52,211,153,.08);
}
.result-panel.danger {
    background:
        linear-gradient(var(--surface), var(--surface)) padding-box,
        linear-gradient(135deg, rgba(251,113,133,.6), rgba(249,115,22,.5), rgba(251,113,133,.4)) border-box;
    border: 2px solid transparent;
    box-shadow:
        0 0 60px rgba(251,113,133,.12),
        0 0 120px rgba(251,113,133,.05),
        0 0 0 1px rgba(251,113,133,.08),
        inset 0 1px 0 rgba(251,113,133,.08);
}
.result-panel .top-bar {
    position: absolute; top: 0; left: 0; right: 0; height: 4px;
}
.result-panel.safe .top-bar {
    background: linear-gradient(90deg, #34d399, #22d3ee, #34d399);
    background-size: 200% 100%;
    animation: shimmer 3s ease infinite;
    box-shadow: 0 0 30px rgba(52,211,153,.5), 0 2px 15px rgba(52,211,153,.3);
}
.result-panel.danger .top-bar {
    background: linear-gradient(90deg, #fb7185, #f97316, #fb7185);
    background-size: 200% 100%;
    animation: shimmer 3s ease infinite;
    box-shadow: 0 0 30px rgba(251,113,133,.5), 0 2px 15px rgba(251,113,133,.3);
}

/* ── Verdict row ──────────────────────────────── */
.verdict-row { display: flex; align-items: center; gap: 16px; }
.verdict-icon {
    width: 64px; height: 64px; border-radius: 20px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem;
    animation: slideUp .5s cubic-bezier(.22,1,.36,1) .1s both;
}
.safe .verdict-icon {
    background: rgba(52,211,153,.10);
    border: 1.5px solid rgba(52,211,153,.25);
    box-shadow: 0 0 40px rgba(52,211,153,.20), inset 0 0 25px rgba(52,211,153,.06);
}
.danger .verdict-icon {
    background: rgba(251,113,133,.10);
    border: 1.5px solid rgba(251,113,133,.25);
    box-shadow: 0 0 40px rgba(251,113,133,.20), inset 0 0 25px rgba(251,113,133,.06);
}
.verdict-text { font-size: 1.6rem; font-weight: 900; letter-spacing: -.5px; }
.safe .verdict-text {
    background: linear-gradient(135deg, #34d399, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    filter: drop-shadow(0 0 20px rgba(52,211,153,.4));
}
.danger .verdict-text {
    background: linear-gradient(135deg, #fb7185, #f97316);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    filter: drop-shadow(0 0 20px rgba(251,113,133,.4));
}
.verdict-sub { font-size: .76rem; color: var(--text2); margin-top: 2px; }

/* ── Gauge (circular ring) ────────────────────── */
.gauge-wrap {
    position: relative;
    width: 240px; height: 240px;
    margin: 1.5rem auto .5rem;
}
.gauge-overlay {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.gauge-value {
    font-size: 3.5rem; font-weight: 900;
    letter-spacing: -2px;
    animation: numReveal .7s cubic-bezier(.22,1,.36,1) .3s both;
}
.safe .gauge-value { color: #34d399; text-shadow: 0 0 50px rgba(52,211,153,.6); }
.danger .gauge-value { color: #fb7185; text-shadow: 0 0 50px rgba(251,113,133,.6); }
.gauge-label {
    font-size: .7rem; font-weight: 700; color: var(--text2);
    text-transform: uppercase; letter-spacing: 2px; margin-top: 2px;
}

/* ── Meta chips ───────────────────────────────── */
.meta-row { display: flex; gap: 10px; margin-top: 1.2rem; flex-wrap: wrap; }
.meta-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background:
        linear-gradient(rgba(167,139,250,.04), rgba(167,139,250,.01)) padding-box,
        linear-gradient(135deg, rgba(167,139,250,.18), rgba(34,211,238,.12)) border-box;
    border: 1px solid transparent;
    border-radius: 10px; padding: 6px 13px;
    font-size: .7rem; font-weight: 600; color: var(--text2);
    backdrop-filter: blur(4px);
}

/* ── Category cards ───────────────────────────── */
.cat-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
    margin-top: .8rem;
}
@media(max-width:640px){ .cat-grid{grid-template-columns:1fr} }
.cat-card {
    position: relative; overflow: hidden;
    background: var(--surface2);
    border: 1.5px solid var(--border);
    border-radius: 20px;
    padding: 22px;
    display: flex; align-items: center; gap: 14px;
    transition: all .3s cubic-bezier(.22,1,.36,1);
    animation: cardEntrance .5s cubic-bezier(.22,1,.36,1) backwards;
}
.cat-card:nth-child(1) { animation-delay: .05s; }
.cat-card:nth-child(2) { animation-delay: .12s; }
.cat-card:nth-child(3) { animation-delay: .19s; }
.cat-card:nth-child(4) { animation-delay: .26s; }
.cat-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 16px 40px rgba(0,0,0,.35);
    border-color: #2a3454;
}
.cat-card.active {
    background:
        linear-gradient(
            color-mix(in srgb, var(--cat-color) 12%, var(--surface2)),
            color-mix(in srgb, var(--cat-color) 4%, var(--surface2))
        ) padding-box,
        linear-gradient(135deg, var(--cat-color), color-mix(in srgb, var(--cat-color) 40%, #22d3ee)) border-box;
    border: 2px solid transparent;
    animation: cardEntrance .5s cubic-bezier(.22,1,.36,1) backwards,
              cardPulse 2s ease-in-out infinite;
}
.cat-card .bg-fill {
    position: absolute; left: 0; top: 0; bottom: 0;
    border-radius: 20px; opacity: .10;
    transition: width .8s cubic-bezier(.22,1,.36,1);
}
.cat-card.active .bg-fill { opacity: .15; }
.cat-card .icon-box {
    width: 56px; height: 56px; border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; z-index: 1; flex-shrink: 0;
    background: color-mix(in srgb, var(--cat-color) 10%, transparent);
    border: 1px solid color-mix(in srgb, var(--cat-color) 15%, transparent);
    transition: all .3s;
}
.cat-card.active .icon-box {
    box-shadow: 0 0 35px color-mix(in srgb, var(--cat-color) 35%, transparent);
    background: color-mix(in srgb, var(--cat-color) 18%, transparent);
    border-color: color-mix(in srgb, var(--cat-color) 30%, transparent);
    animation: iconPulse 2s ease-in-out infinite;
}
.cat-card .info { flex: 1; z-index: 1; min-width: 0; }
.cat-card .cat-name { font-weight: 700; font-size: .88rem; color: var(--text); }
.cat-card.active .cat-name { color: #fff; }
.cat-card .cat-desc { font-size: .66rem; color: var(--text2); margin-top: 2px; }
.cat-card .bar-track {
    height: 7px; background: rgba(255,255,255,.06); border-radius: 4px;
    margin-top: 10px; overflow: hidden;
    box-shadow: inset 0 1px 2px rgba(0,0,0,.3);
}
.cat-card .bar-fill {
    height: 100%; border-radius: 4px;
    transform-origin: left;
    animation: barGrow .8s cubic-bezier(.22,1,.36,1);
    box-shadow: 0 0 14px currentColor;
}
.cat-card.active .bar-fill {
    box-shadow: 0 0 20px currentColor, 0 0 40px currentColor;
}
.cat-card .cat-pct {
    font-size: 1.2rem; font-weight: 800; z-index: 1;
    min-width: 50px; text-align: right; color: var(--text2);
    transition: color .3s;
}
.cat-card.active .cat-pct {
    color: var(--cat-color);
    text-shadow: 0 0 30px color-mix(in srgb, var(--cat-color) 60%, transparent);
    font-size: 1.45rem;
}
/* DETECTED badge */
.cat-badge {
    position: absolute; top: 10px; right: 12px;
    font-size: .58rem; font-weight: 800; letter-spacing: 1.5px;
    text-transform: uppercase; z-index: 3;
    padding: 3px 10px; border-radius: 6px;
    animation: badgeBlink 1.5s ease-in-out infinite;
}
.cat-card.active .cat-badge {
    color: var(--cat-color);
    background: color-mix(in srgb, var(--cat-color) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--cat-color) 25%, transparent);
    box-shadow: 0 0 16px color-mix(in srgb, var(--cat-color) 25%, transparent);
}
.cat-card .thr-line {
    position: absolute; top: 0; bottom: 0; width: 2px;
    background: var(--text2); opacity: .25; z-index: 2;
}
.cat-card .thr-tag { font-size: .56rem; color: var(--text2); margin-top: 4px; }

/* ── Footer ───────────────────────────────────── */
.app-footer {
    margin-top: 3.5rem; padding: 2rem 1.5rem;
    text-align: center;
    font-size: .72rem; color: var(--text2); line-height: 1.8;
    position: relative;
}
.app-footer::before {
    content: ''; display: block; height: 1px; margin-bottom: 2rem;
    background: linear-gradient(90deg, transparent 5%, #a78bfa 30%, #22d3ee 50%, #34d399 70%, transparent 95%);
    opacity: .5;
}
.app-footer strong { color: #fbbf24; }
.app-footer .metrics {
    display: inline-flex; gap: 16px; margin-top: 8px;
    font-size: .68rem; font-weight: 600;
}
.app-footer .metrics span { color: var(--accent); }

/* ── Streamlit overrides ──────────────────────── */
[data-testid="stAlert"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
}
[data-testid="stTabs"] { background: transparent !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--surface) !important;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 5px;
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text2) !important;
    border-radius: 11px !important;
    font-weight: 600 !important;
    font-size: .8rem !important;
    padding: 10px 18px !important;
    border: none !important;
    transition: all .25s;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #a78bfa, #7c3aed) !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(167,139,250,.3), inset 0 1px 0 rgba(255,255,255,.1);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabs"] [data-testid="stTabContent"] { padding-top: 1rem !important; }

.rng-info, .gen-info {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 20px;
    font-size: .82rem;
    color: var(--text2);
    line-height: 1.65;
    margin-bottom: .8rem;
}
.rng-info strong, .gen-info strong { color: var(--text); }
.gen-info code {
    background: rgba(167,139,250,.08); color: #a78bfa;
    padding: 2px 10px; border-radius: 6px; font-size: .76rem;
    border: 1px solid rgba(167,139,250,.12);
}
.rng-badge {
    display: inline-block; padding: 3px 12px; border-radius: 8px;
    font-size: .68rem; font-weight: 700; margin-right: 4px;
}
.rng-badge.safe  { background: rgba(52,211,153,.10); color: #34d399; border: 1px solid rgba(52,211,153,.15); }
.rng-badge.toxic { background: rgba(251,113,133,.10); color: #fb7185; border: 1px solid rgba(251,113,133,.15); }

.gen-result {
    background:
        linear-gradient(var(--surface), var(--surface)) padding-box,
        linear-gradient(135deg, rgba(167,139,250,.4), rgba(34,211,238,.3)) border-box;
    border: 1.5px solid transparent;
    border-radius: 16px;
    padding: 20px 22px;
    margin: 1rem 0;
    animation: slideUp .35s cubic-bezier(.22,1,.36,1);
    box-shadow: 0 0 40px rgba(167,139,250,.08);
}
.gen-result .gen-head {
    font-size: .68rem; font-weight: 700; color: #a78bfa;
    text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px;
}
.gen-result .gen-text {
    font-size: 1.05rem; color: var(--text); line-height: 1.65; font-weight: 500;
}
.gen-result .gen-meta { margin-top: 12px; font-size: .68rem; color: var(--text2); }

.gen-pre-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
    margin-top: .6rem;
}
@media(max-width:640px){ .gen-pre-grid{grid-template-columns:1fr} }
.gen-pre-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    font-size: .8rem;
    color: var(--text);
    line-height: 1.5;
    cursor: pointer;
    transition: all .25s;
}
.gen-pre-card:hover {
    border-color: rgba(167,139,250,.3);
    background: rgba(167,139,250,.04);
    transform: translateY(-1px);
}
.gen-pre-card .gen-pre-label {
    font-size: .64rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .8px; margin-bottom: 5px;
}
.gen-pre-card .gen-pre-label.mis { color: #fb7185; }
.gen-pre-card .gen-pre-label.safe { color: #34d399; }

[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] label {
    color: var(--text2) !important;
    font-size: .72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* ── Animations ───────────────────────────────── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: none; }
}
@keyframes barGrow { from { transform: scaleX(0); } }
@keyframes numReveal {
    from { opacity: 0; transform: translateY(12px) scale(.8); filter: blur(8px); }
    to   { opacity: 1; transform: none; filter: blur(0); }
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes badgeFloat {
    0%, 100% { transform: translateY(0) scale(1); }
    50%      { transform: translateY(-8px) scale(1.02); }
}
@keyframes badgeGlow {
    0%, 100% { box-shadow: 0 8px 50px rgba(167,139,250,.3), 0 0 100px rgba(167,139,250,.12), 0 0 0 8px rgba(167,139,250,.06); }
    50%      { box-shadow: 0 8px 60px rgba(167,139,250,.45), 0 0 120px rgba(167,139,250,.2), 0 0 0 14px rgba(167,139,250,.10); }
}
@keyframes orbDrift1 {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(-40px, 30px) scale(1.1); }
}
@keyframes orbDrift2 {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(30px, -25px) scale(1.08); }
}
@keyframes orbDrift3 {
    0%   { transform: translate(0, 0) scale(1); opacity: .6; }
    100% { transform: translate(-20px, 15px) scale(.9); opacity: 1; }
}
@keyframes scanMove {
    0%   { top: 10%; opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 90%; opacity: 0; }
}
@keyframes gradShift {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
@keyframes heroBorder {
    0%   { background-position: 0 0, 0% 0; }
    100% { background-position: 0 0, -300% 0; }
}
@keyframes cardEntrance {
    from { opacity: 0; transform: translateY(24px) scale(.95); }
    to   { opacity: 1; transform: none; }
}
@keyframes cardPulse {
    0%, 100% {
        box-shadow:
            0 0 40px color-mix(in srgb, var(--cat-color) 20%, transparent),
            0 0 80px color-mix(in srgb, var(--cat-color) 8%, transparent),
            0 8px 24px rgba(0,0,0,.3);
    }
    50% {
        box-shadow:
            0 0 60px color-mix(in srgb, var(--cat-color) 35%, transparent),
            0 0 120px color-mix(in srgb, var(--cat-color) 15%, transparent),
            0 8px 30px rgba(0,0,0,.35);
    }
}
@keyframes iconPulse {
    0%, 100% {
        box-shadow: 0 0 35px color-mix(in srgb, var(--cat-color) 35%, transparent);
        transform: scale(1);
    }
    50% {
        box-shadow: 0 0 50px color-mix(in srgb, var(--cat-color) 50%, transparent);
        transform: scale(1.06);
    }
}
@keyframes badgeBlink {
    0%, 100% { opacity: 1; }
    50%      { opacity: .4; }
}
</style>
""")


# ── SVG gauge helper ──────────────────────────────────────────────────────────
def gauge_svg(pct, c1, c2, size=240):
    r, sw = 96, 12
    cx, cy = size // 2, size // 2
    circ = 2 * math.pi * r
    dash = circ * max(pct, 0.01)
    gap = circ - dash
    gid = f"g{abs(hash((pct, c1))) % 99999}"
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}"
         style="transform:rotate(-90deg);filter:drop-shadow(0 0 20px {c1}50) drop-shadow(0 0 40px {c1}25);">
      <defs>
        <linearGradient id="{gid}" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="{c1}"/>
          <stop offset="100%" stop-color="{c2}"/>
        </linearGradient>
      </defs>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
              stroke="rgba(255,255,255,0.04)" stroke-width="{sw}"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
              stroke="url(#{gid})" stroke-width="{sw}"
              stroke-linecap="round" stroke-dasharray="{dash:.1f} {gap:.1f}"/>
    </svg>"""


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    _tok = AutoTokenizer.from_pretrained(BERT_MODEL)
    _mdl = MultiTaskModel(BERT_MODEL, n_ml_labels=len(LABEL_COLS))
    _mdl.load_state_dict(
        torch.load(
            os.path.join(SAVE_DIR, "multitask_model_weights.pt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    _mdl.eval()
    with open(os.path.join(SAVE_DIR, "ml_thresholds.json"), encoding="utf-8") as f:
        _thr = json.load(f)
    return _tok, _mdl, _thr


@torch.no_grad()
def predict(text, tok_arg, mdl_arg, thr_arg):
    t0 = time.perf_counter()
    enc = tok_arg(text, return_tensors="pt", truncation=True, padding=True, max_length=BERT_MAX_LEN)
    bl, ml = mdl_arg(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    elapsed = (time.perf_counter() - t0) * 1000
    bin_prob = torch.softmax(bl, dim=-1)[0, 1].item()
    bin_pred = int(bl.argmax(1))
    ml_probs = torch.sigmoid(ml)[0].tolist()
    ml_preds = (
        [int(p > t) for p, t in zip(ml_probs, thr_arg)] if bin_pred == 1 else [0] * len(thr_arg)
    )
    return bin_pred, bin_prob, ml_probs, ml_preds, elapsed


tok, mdl, thr = load_model()

# ━━━━━━━━━━━━━━━━━━  U I  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Hero banner ───────────────────────────────────────────────────────────────
st.html("""
<div class="hero">
    <div class="hero-dots"></div>
    <div class="hero-bg">
        <div class="orb orb1"></div>
        <div class="orb orb2"></div>
        <div class="orb orb3"></div>
    </div>
    <div class="hero-scan"></div>
    <div class="hero-badge">🛡️</div>
    <h1>Misogyny <span class="grad">Classifier</span></h1>
    <div class="tagline">
        Multi-Task RoBERTa · FGM adversarial training<br>
        Cascaded binary &rarr; multi-label pipeline &middot; MAMI dataset
    </div>
    <div class="hero-stats">
        <div class="hero-stat"><span class="stat-val">.825</span><span class="stat-lbl">Binary F1</span></div>
        <div class="hero-stat"><span class="stat-val">.520</span><span class="stat-lbl">ML F1</span></div>
        <div class="hero-stat"><span class="stat-val">7.5K</span><span class="stat-lbl">Samples</span></div>
        <div class="hero-stat"><span class="stat-val">4</span><span class="stat-lbl">Categories</span></div>
    </div>
    <div class="pills">
        <span class="pill">🧠 twitter-roberta-base-hate</span>
        <span class="pill">⚔️ FGM adversarial</span>
        <span class="pill">📊 Cascaded pipeline</span>
        <span class="pill">🎯 Optimised thresholds</span>
    </div>
</div>
""")

# ── Tabs: Examples / Random / Generator / Write ──────────────────────────────
tab_examples, tab_random, tab_gen, tab_write = st.tabs(
    ["📝 Examples", "🎲 Random", "🤖 AI Generator", "✏️ Write"]
)

with tab_examples:
    st.html('<div class="sec-label">Safe</div>')
    for i, txt in enumerate(EXAMPLES_SAFE):
        if st.button(f"🟢 {txt}", key=f"se{i}", use_container_width=True):
            st.session_state["txt"] = txt

    st.html('<div class="sec-label" style="margin-top:.8rem">Misogynous (one per category)</div>')
    cat_labels = ["Shaming", "Stereotype", "Objectification", "Violence"]
    cols_t = st.columns(2)
    for i, txt in enumerate(EXAMPLES_TOXIC):
        with cols_t[i % 2]:
            lbl = f"🔴 [{cat_labels[i]}] {txt[:34]}{'…' if len(txt)>34 else ''}"
            if st.button(lbl, key=f"te{i}", use_container_width=True):
                st.session_state["txt"] = txt

with tab_random:
    ds = load_dataset_texts()
    n_safe, n_mis = len(ds["safe"]), len(ds["misogynous"])
    st.html(f"""
    <div class="rng-info">
        Pull a random caption from the <strong>MAMI training set</strong> and classify it live.<br>
        <span class="rng-badge safe">{n_safe} safe</span>
        <span class="rng-badge toxic">{n_mis} misogynous</span>
    </div>""")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        if st.button("🎲  Random (any)", key="rng_any", use_container_width=True):
            pool = ds["safe"] + ds["misogynous"]
            if pool:
                st.session_state["txt"] = random.choice(pool)
    with rc2:
        if st.button("🟢  Random safe", key="rng_safe", use_container_width=True):
            if ds["safe"]:
                st.session_state["txt"] = random.choice(ds["safe"])
    with rc3:
        if st.button("🔴  Random toxic", key="rng_toxic", use_container_width=True):
            if ds["misogynous"]:
                st.session_state["txt"] = random.choice(ds["misogynous"])

with tab_gen:
    st.html("""
    <div class="gen-info">
        <strong>RAG Meme Generator</strong> — uses Mistral 7B + retrieval-augmented generation
        to create new meme captions on the fly.<br>
        Requires the Docker stack to be running
        (<code>docker compose up</code>).
    </div>""")

    # ── Live generation ───────────────────────────────────────────────────
    st.html('<div class="sec-label">Generate live with Mistral 7B</div>')
    gc1, gc2 = st.columns([2, 1])
    with gc1:
        live_cat = st.selectbox(
            "Type",
            ["neutral", "shaming", "stereotype", "objectification", "violence"],
            key="live_cat",
        )
    with gc2:
        st.write("")  # spacer
        gen_go = st.button("⚡ Generate", key="gen_live", use_container_width=True)

    if gen_go:
        try:
            with st.spinner("Generating with Mistral 7B…"):
                meme_text, used_rag = generate_meme_live(live_cat)
            if meme_text:
                st.session_state["txt"] = meme_text
                rag_info = "RAG context ✓" if used_rag else "No RAG (service offline)"
                st.html(f"""
                <div class="gen-result">
                    <div class="gen-head">Generated · {live_cat}</div>
                    <div class="gen-text">{meme_text}</div>
                    <div class="gen-meta">{rag_info} · Mistral 7B Q4_K_M</div>
                </div>""")
            else:
                st.warning("Generator returned empty output. Try again.")
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to llama.cpp. "
                "Start the Docker stack with `docker compose up` to enable live generation."
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Generation failed: {exc}")
st.html('<div class="sec-label" style="margin-top:1.2rem">Text to classify</div>')
user_text = st.text_area(
    "input",
    value=st.session_state.get("txt", ""),
    height=110,
    placeholder="Pick an example, hit Random, generate one, or type your own…",
    label_visibility="collapsed",
)
go = st.button("Classify", type="primary", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if go and user_text.strip():
    with st.spinner("Analysing…"):
        bp, bprob, mprobs, mpreds, ms = predict(user_text.strip(), tok, mdl, thr)

    is_safe = bp == 0
    conf = (1 - bprob) if is_safe else bprob
    panel_cls = "safe" if is_safe else "danger"
    verdict_icon = "✅" if is_safe else "⚠️"
    verdict_title = "Not Misogynous" if is_safe else "Misogynous"
    c_main = "#34d399" if is_safe else "#f87171"
    c_end = "#059669" if is_safe else "#dc2626"
    svg = gauge_svg(conf, c_main, c_end)

    panel_html = f"""
    <div class="result-panel {panel_cls}">
        <div class="top-bar"></div>
        <div class="verdict-row">
            <div class="verdict-icon">{verdict_icon}</div>
            <div>
                <div class="verdict-text">{verdict_title}</div>
                <div class="verdict-sub">Binary classification result</div>
            </div>
        </div>
        <div class="gauge-wrap">
            {svg}
            <div class="gauge-overlay">
                <div class="gauge-value" style="color:{c_main}">{conf:.1%}</div>
                <div class="gauge-label">Confidence</div>
            </div>
        </div>
        <div class="meta-row">
            <div class="meta-chip">⚡ {ms:.0f} ms</div>
            <div class="meta-chip">🧠 RoBERTa-hate</div>
            <div class="meta-chip">📏 {len(user_text.split())} tokens</div>
        </div>
    </div>"""
    st.html(panel_html)

    # ── Category breakdown (only when misogynous) ─────────────────────────
    if not is_safe:
        st.html('<div class="sec-label" style="margin-top:.6rem">Category breakdown</div>')
        cards = '<div class="cat-grid">'
        for label, prob, is_detected, t in zip(LABEL_COLS, mprobs, mpreds, thr):
            m = LABEL_META[label]
            act = "active" if is_detected else ""
            grad = f"linear-gradient(135deg,{m['grad'][0]},{m['grad'][1]})"
            status = "DETECTED" if is_detected else "below threshold"
            badge = '<div class="cat-badge">DETECTED</div>' if is_detected else ""
            cards += f"""
            <div class="cat-card {act}" style="--cat-color:{m['color']}">
                {badge}
                <div class="bg-fill" style="width:{prob*100:.0f}%;background:{grad}"></div>
                <div class="thr-line" style="left:{t*100:.0f}%" title="threshold {t:.2f}"></div>
                <div class="icon-box">{m['icon']}</div>
                <div class="info">
                    <div class="cat-name">{label.capitalize()}</div>
                    <div class="cat-desc">{LABEL_DESC[label]}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{prob*100:.0f}%;background:{grad}"></div>
                    </div>
                    <div class="thr-tag">prob {prob:.2f} vs thr {t:.2f} · {status}</div>
                </div>
                <div class="cat-pct">{prob:.0%}</div>
            </div>"""
        cards += "</div>"
        st.html(cards)

elif go:
    st.warning("Please enter some text to classify.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.html("""
<div class="app-footer">
    <strong>⚠️ Limitation</strong> — Trained on meme captions (MAMI dataset). May over-trigger
    on formal text mentioning gender. Best results on short, meme-style text.
    <div class="metrics">
        <span>Binary F1 0.8253</span> · <span>ML E2E F1 0.5199</span> ·
        Thresholds optimised on validation set
    </div>
</div>
""")
