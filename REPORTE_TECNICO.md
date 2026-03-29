# Technical Report — Misogyny Detection in Memes

**Capstone Project · Natural Language Processing**  
**Professor:** Javier Lasheras Navas  
**Dataset:** MAMI (Multimedia Automatic Misogyny Identification)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Related Work](#2-related-work)
3. [Dataset and Preprocessing](#3-dataset-and-preprocessing)
4. [Data Split Strategy](#4-data-split-strategy)
5. [Non-DL Baseline: TF-IDF + Logistic Regression](#5-non-dl-baseline-tf-idf--logistic-regression)
6. [Transformer Model: Selection and Justification](#6-transformer-model-selection-and-justification)
7. [Multi-Task Learning Architecture (MTL)](#7-multi-task-learning-architecture-mtl)
8. [Adversarial Training: FGM](#8-adversarial-training-fgm)
9. [Optimization Strategy](#9-optimization-strategy)
10. [Cascaded Pipeline: End-to-End Evaluation](#10-cascaded-pipeline-end-to-end-evaluation)
11. [Per-Label Threshold Tuning](#11-per-label-threshold-tuning)
12. [Error Analysis](#12-error-analysis)
13. [RAG Pipeline: Meme Generation](#13-rag-pipeline-meme-generation)
14. [Infrastructure and Deployment](#14-infrastructure-and-deployment)
15. [Streamlit Demo](#15-streamlit-demo)
16. [ONNX Export](#16-onnx-export)
17. [Code Quality and Tests](#17-code-quality-and-tests)
18. [Results Summary](#18-results-summary)
19. [Conclusions and Future Work](#19-conclusions-and-future-work)

---

## 1. Project Overview

### Motivation and Context

Online misogyny — hostile or derogatory content targeting women — has become one of the most prevalent forms of hate speech on social media. Internet memes are a particularly insidious vector because they combine visual humor with textual messages, allowing harmful stereotypes or threats to spread virally under the guise of "just a joke." Automated detection is essential for content moderation at scale, but presents unique NLP challenges: informal register, deliberate misspellings, irony, sarcasm, and implicit meaning that only becomes harmful when combined with the image.

The **MAMI (Multimedia Automatic Misogyny Identification)** shared task, organized at SemEval-2022 (Task 5), addresses exactly this problem. It defines two sub-tasks:

- **Task A (Binary):** Determine whether a meme is misogynous or not.
- **Task B (Multi-label):** For misogynous memes, identify which sub-types of misogyny are present: *shaming*, *stereotype*, *objectification*, and/or *violence*.

This project tackles both sub-tasks using a **text-only** approach on OCR-extracted captions from the memes. While this necessarily loses visual context (which contributes to meaning in many memes), it allows us to focus on the NLP challenges and demonstrates that significant classification performance can be achieved from text alone.

### Project Architecture

The project consists of **two complementary stages**:

| Stage | Description | Purpose |
|-------|-------------|---------|
| **Stage 1 — RAG Meme Generator** | Retrieval-Augmented Generation pipeline using a locally quantized LLM (Mistral 7B Instruct) to generate misogynous meme captions across 4 categories + neutral. | Augments the evaluation dataset with synthetic examples and demonstrates generation capabilities. |
| **Stage 2 — Misogyny Classifier** | Cascaded multi-task classifier that first detects whether a meme is misogynous (binary) and, if positive, identifies the active sub-types (multi-label). | Core classification system — the main deliverable of the project. |

The two stages are connected: the RAG pipeline (Stage 1) generates synthetic memes that serve as an additional evaluation benchmark for the classifier (Stage 2), testing generalization to LLM-generated content.

The complete system is orchestrated with **Docker Compose** (5 services) and includes an **interactive Streamlit demo** for real-time classification.

### Design Decisions and Rationale

Several high-level architectural choices shape the project:

1. **Cascaded pipeline over flat multi-label**: Rather than predicting all 5 labels (binary + 4 sub-types) in a single pass, we use a two-stage cascade. This reflects the logical dependency: sub-type labels are only meaningful when a meme is misogynous. The cascade prevents incoherent predictions (e.g., a meme classified as "not misogynous" but tagged with "violence").

2. **Multi-Task Learning (MTL) over separate models**: Instead of training two independent models (one for binary, one for multi-label), we use a shared backbone with two task-specific heads. This allows the backbone to benefit from gradients of both tasks, effectively using all 7,500 samples for representation learning even though only ~2,655 are misogynous.

3. **Text-only approach**: While the MAMI task provides images, we intentionally focus on OCR text to explore the limits of NLP-only methods. This provides a clean baseline that could later be extended with multimodal features (CLIP, ViLT).

4. **Domain-specific pre-trained model**: Instead of a general-purpose BERT, we select a RoBERTa variant pre-trained on 124M tweets and fine-tuned on hate speech data. Domain match between pre-training and target data is critical for downstream performance (§6).

### Project Structure

```
.
├── src/
│   ├── MultiClassifier.ipynb     # Main notebook (full pipeline)
│   ├── app.py                   # Streamlit demo
│   ├── backend/
│   │   ├── flask_app/           # Flask REST API (RAG)
│   │   │   ├── agent.py         # ReAct agent (Bonus)
│   │   │   ├── app.py           # REST endpoints
│   │   │   ├── chunking.py      # Chunking strategies
│   │   │   ├── config.py        # Centralized configuration
│   │   │   ├── generation.py    # Generation with llama.cpp
│   │   │   ├── ingestion.py     # PDF/DOCX parsing
│   │   │   └── retrieval.py     # Dense + BM25 + Hybrid (RRF)
│   │   └── scripts/
│   │       ├── upload_training_to_rag.py
│   │       ├── generate_memes_rag.py
│   │       └── csvapdf.py
│   ├── frontend/streamlit/      # Streamlit for RAG
│   ├── data/training/           # training.csv (7,500 samples)
│   ├── models/                  # GGUF weights / trained/
│   └── evaluation/results/      # Evaluation CSVs and JSONs
├── tests/                       # 61 unit tests (pytest)
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

---

## 2. Related Work

### Hate Speech and Misogyny Detection

The automatic detection of hate speech and misogyny in online content has attracted growing research attention. The SemEval-2022 Task 5 — MAMI (Fersini et al., 2022) established a benchmark for multimedia misogyny identification that attracted 83 teams internationally. Top-performing systems combined visual and textual features, with the best text-only approaches achieving F1-macro scores around 0.71–0.73 on Task A (binary). Our work builds on this shared task's dataset and evaluation protocol, extending it with multi-task learning and adversarial training techniques not explored by most participating teams.

### Transfer Learning and Domain-Specific Pre-training

The dominant paradigm in modern NLP is transfer learning from pre-trained language models (PLMs). BERT (Devlin et al., 2019) demonstrated the effectiveness of bidirectional masked language modeling, followed by RoBERTa (Liu et al., 2019) which improved training methodology through dynamic masking, larger batches, and removal of the Next Sentence Prediction objective. Crucially, Gururangan et al. (2020) showed in "Don't Stop Pretraining" that **domain-adaptive pre-training** — continued pre-training on in-domain data — consistently improves downstream performance, even outperforming larger general-domain models. This finding directly motivates our choice of `cardiffnlp/twitter-roberta-base-hate`, a RoBERTa model continually pre-trained on 124M tweets and fine-tuned on hate speech datasets (HatEval, OffComEval).

### Multi-Task Learning

Multi-Task Learning (MTL) was formalized by Caruana (1997) as a paradigm where related tasks share a representation, using the inductive bias from auxiliary tasks as a regularizer. Ruder (2017) provided a comprehensive survey of MTL in deep learning, identifying mechanisms through which MTL helps: implicit data augmentation, attention focusing, eavesdropping, and representation bias. In NLP, MTL has been successfully applied to joint sentiment analysis (Akhtar et al., 2019), named entity recognition and relation extraction (Eberts & Ulges, 2020), and hate speech detection with emotion classification (Plaza-del-Arco et al., 2021). Our architecture follows the **hard parameter sharing** approach — a shared encoder with task-specific heads — which Ruder (2017) identifies as the most common and effective MTL architecture for related tasks.

### Adversarial Training for NLP

Adversarial training was introduced by Goodfellow et al. (2015) to improve model robustness against small input perturbations. Since discrete text tokens cannot be directly perturbed, Miyato et al. (2017) proposed applying perturbations to **continuous word embeddings** instead, achieving significant improvements on semi-supervised text classification. The Fast Gradient Method (FGM) computes a single perturbation step in the gradient direction, providing a favorable accuracy–efficiency trade-off. More recent methods like PGD (Madry et al., 2018), FreeLB (Zhu et al., 2020), and SMART (Jiang et al., 2020) use multi-step perturbations for stronger adversarial examples, but at 2–5× the computational cost. We adopt FGM for its simplicity and negligible overhead (~50% training time increase for +1–3% F1 improvement).

### Retrieval-Augmented Generation

Lewis et al. (2020) introduced Retrieval-Augmented Generation (RAG), combining a non-parametric retriever with a parametric generator to ground LLM outputs in retrieved evidence. For retrieval, hybrid approaches combining dense semantic search with sparse lexical methods (BM25) have proven more robust than either alone (Cormack et al., 2009 on Reciprocal Rank Fusion). The ReAct framework (Yao et al., 2023) extends RAG with a reasoning-action loop, enabling multi-step retrieval for complex queries. Our RAG pipeline integrates these approaches: hybrid retrieval (Dense + BM25 + RRF), a locally quantized Mistral 7B Instruct model, and an optional ReAct agent for iterative retrieval.

### Positioning of This Work

Our project makes several contributions relative to the existing literature:
- Unlike most SemEval-2022 Task 5 participants who used separate binary and multi-label models, we employ **multi-task learning** with a shared backbone, enabling the multi-label head to benefit from all 7,500 samples.
- We apply **FGM adversarial training** to hate speech classification, a technique more commonly used in general NLU benchmarks but underexplored in the misogyny detection domain.
- We combine the classifier with a **complete RAG pipeline** for synthetic meme generation, creating an end-to-end system that both classifies and generates content for evaluation.
- We provide a **thorough error analysis** with linguistic categorization of failure modes, going beyond standard metrics to explain *why* the model fails.

---

## 3. Dataset and Preprocessing

### MAMI Dataset

The MAMI dataset originates from the SemEval-2022 Task 5 shared challenge, which attracted 83 participating teams internationally. It was curated from real memes collected from social media platforms, with expert annotation for misogyny categories.

| Property | Value | Notes |
|----------|-------|-------|
| **Source** | SemEval-2022 Task 5 — MAMI | International shared task on misogyny identification |
| **Format** | TSV (tab-separated) | One row per meme |
| **Total Samples** | 7,500 | Used in full (no filtering) |
| **Language** | English (informal, internet slang) | Includes abbreviations, typos, sarcasm, emoji references |
| **Class Balance (binary)** | ~47% misogynous / ~53% non-misogynous | Moderate imbalance requiring weighted losses |
| **Multi-label** | 4 sub-categories, non-exclusive | One meme can carry multiple labels simultaneously |

### Columns and Label Semantics

| Column | Type | Description | Prevalence (approx.) |
|--------|------|-------------|---------------------|
| `file_name` | str | Meme image identifier | — |
| `misogynous` | int (0/1) | 1 if the meme is misogynous | ~47% positive |
| `shaming` | int (0/1) | Body shaming or slut shaming — attacks targeting physical appearance, sexual behavior, or personal capabilities | ~27% of misogynous |
| `stereotype` | int (0/1) | Reinforcement of gender stereotypes — women belong in the kitchen, are bad drivers, are overly emotional, etc. | ~61% of misogynous (most frequent) |
| `objectification` | int (0/1) | Sexual objectification — reducing women to their physical attributes or treating them as objects for male pleasure | ~46% of misogynous |
| `violence` | int (0/1) | Threats of physical or psychological violence — explicit or implicit threats, celebration of violence against women | ~21% of misogynous (rarest) |
| `Text Transcription` | str | OCR-extracted text from the meme image | Variable quality; some near-empty |

### Label Co-occurrence and Multi-label Complexity

A critical characteristic of the MAMI dataset is that the 4 sub-type labels are **not mutually exclusive**. A single misogynous meme can simultaneously exhibit *stereotype* and *objectification*, or *shaming* and *violence*. This multi-label nature means:

- **The label space has 2⁴ = 16 possible combinations** (for misogynous memes), though not all are equally represented.
- **Most common pattern**: `stereotype` alone or `stereotype + objectification`.
- **Rarest pattern**: `violence` alone — violence is almost always accompanied by at least one other category.
- **Implication for modeling**: Binary relevance (independent per-label predictions) is a reasonable first approach, but ignores label correlations. Our MTL architecture (§7) addresses this partially through the shared backbone.

### Label Distribution Among Misogynous Samples

| Label | Approx. Count (of ~3,530 misogynous) | Proportion | Pos/Neg Ratio |
|-------|--------------------------------------|------------|---------------|
| **stereotype** | ~2,161 | ~61% | 1.57:1 |
| **objectification** | ~1,625 | ~46% | 0.85:1 |
| **shaming** | ~953 | ~27% | 0.37:1 |
| **violence** | ~741 | ~21% | 0.27:1 |

This severe imbalance across labels explains two key design decisions:
1. **Per-label `pos_weight`** in the BCE loss function (§7) to compensate for under-represented categories.
2. **Per-label threshold tuning** (§11) instead of a uniform 0.5 threshold, because the optimal decision boundary differs dramatically depending on class prevalence.

### Preprocessing

The preprocessing pipeline is intentionally **minimal**, relying on the transformer tokenizer to handle text normalization:

1. **Text concatenation**: All text columns beyond the fixed label columns are concatenated into a single `Text` field: `df['Text'] = df[text_cols].astype(str).agg(' '.join, axis=1)`. This captures all available textual information from the OCR.

2. **No stemming, lemmatization, or stop-word removal**: Traditional NLP preprocessing is deliberately omitted because the RoBERTa tokenizer uses **Byte-Pair Encoding (BPE)** with a vocabulary of 50,265 sub-word tokens. BPE naturally handles morphological variations (e.g., "running" → "run" + "ning") and is robust to misspellings common in meme text. Aggressive preprocessing (like stemming) can actually *harm* transformer-based models by destroying sub-word information that the tokenizer relies on.

3. **No text cleaning or filtering**: We do not remove special characters, URLs, or hashtags because the model was pre-trained on tweets where these elements carry signal. For example, "#MeToo" or hashtag-style markers are common in both the pre-training data and the target memes.

4. **Sample selection**: The first 7,500 samples are used: `df = df.iloc[:N_EXAMPLES]`. This corresponds to the official MAMI training partition.

---

## 4. Data Split Strategy

### Design Principles

The data split follows three critical principles that ensure experimental rigor:

1. **Single shared split**: Both tasks (binary and multi-label) use **the exact same train/val/test partition**. This is essential for two reasons: (a) it prevents data leakage between tasks (a sample in the binary test set cannot appear in the multi-label training set), and (b) it ensures that metrics are **directly comparable** — any improvement from the transformer over the baseline is measured on identical test samples.

2. **Stratification by binary label**: Because the binary classes are imbalanced (~47% positive, ~53% negative), a purely random split could produce partitions with significantly different class proportions, introducing variance in evaluation. Stratification guarantees that each partition preserves the global binary class ratio, making cross-partition comparisons valid.

3. **Strict temporal discipline**: The test set is never used for any training, threshold tuning, or hyperparameter selection. It is held out exclusively for final evaluation. All intermediate decisions (early stopping, threshold search) are made on the validation set.

### Split Proportions and Rationale

| Partition | % | Samples | Purpose |
|-----------|---|---------|---------|
| **Train** | ~70% | 5,250 | Model training (both binary + multi-label gradients) |
| **Val** | ~10% | 750 | Early stopping criterion + per-label threshold tuning |
| **Test** | 20% | 1,500 | Final held-out evaluation (never seen during training) |

**Why 70/10/20?** The split is implemented as a two-step stratified process:

```python
# Step 1: 80% train+val / 20% test
X_tv, X_test = train_test_split(..., test_size=0.20, stratify=y_bin_all)
# Step 2: 87.5% train / 12.5% val (12.5% of 80% = 10% of total)
X_train, X_val = train_test_split(X_tv, test_size=0.125, stratify=y_bin_tv)
```

The 20% test set (1,500 samples) provides a large enough evaluation set for reliable F1 estimation. The 10% validation set (750 samples) is a compromise: large enough for meaningful early stopping decisions and threshold tuning, but not so large that it starves the training set. Scikit-learn's `train_test_split` with `random_state=42` ensures full reproducibility.

### Specialized Subsets for Multi-label Training

In addition to the main partition, two subsets are derived by **filtering only samples where `misogynous = 1`**:

| Subset | Derived From | Approx. Size | Use |
|--------|-------------|--------------|-----|
| **ML train (mis-only)** | Train | ~2,655 | Computation of per-label `pos_weight` for BCE loss; direct gradient signal for the multi-label head |
| **ML val (mis-only)** | Val | ~379 | Grid-search for optimal per-label thresholds (§11) |

**Why filter misogynous-only for thresholds?** In the cascaded pipeline (§10), the multi-label head is only invoked for samples predicted as misogynous by the binary head. If we tuned thresholds on the full validation set (including non-misogynous samples with `[0,0,0,0]` labels), the abundance of true negatives would bias thresholds upward, rewarding a classifier that simply predicts `[0,0,0,0]` for everything. By tuning on misogynous-only samples, we optimize for the actual decision boundary the model faces at inference time.

**Note on multi-task training**: Although the misogynous-only subsets are used for `pos_weight` computation and threshold tuning, the multi-task training loop (§7) feeds **all 7,500 samples** through the backbone. The binary loss is computed on all samples. The multi-label BCE loss is computed only on the misogynous subset (`mis = blab == 1; loss_ml = ml_loss_fn(ml_logits[mis], mlab[mis])`). This design ensures the backbone receives gradient signal from the entire dataset through the binary task, while the multi-label head focuses exclusively on the informative subset.

---

## 5. Non-DL Baseline: TF-IDF + Logistic Regression

### Purpose of a Non-DL Baseline

Every strong experimental setup requires a **non-deep-learning baseline** for two fundamental reasons:

1. **Value demonstration**: It quantifies the *marginal value* of the transformer model. Without a baseline, we cannot distinguish whether high F1 is due to the model's capabilities or to an inherently easy task. The gain over the baseline (∆F1) is the true measure of the transformer's contribution.

2. **Cost-benefit analysis**: A baseline establishes the cost-effectiveness threshold. If TF-IDF+LR achieved 0.82 F1 while the transformer only reaches 0.83, the 125M-parameter transformer may not justify its computational cost. The larger the gap, the stronger the justification for the DL approach.

### Why TF-IDF + Logistic Regression?

This combination is the **de facto standard baseline** for text classification across the NLP literature (including SemEval shared tasks) for several reasons:

- **TF-IDF** (Term Frequency–Inverse Document Frequency) converts text into a sparse vector where each dimension represents a token weighted by its informativeness. The `sublinear_tf=True` option applies logarithmic dampening (`1 + log(tf)`) to prevent very frequent words from dominating the representation. Combined with `ngram_range=(1,2)`, it captures both individual words and two-word expressions that are often discriminative (e.g., "shut up", "stay home", "only good").

- **Logistic Regression** is a linear classifier that finds the optimal hyperplane separating classes. Despite its simplicity, it is a strong baseline because: (a) it converges reliably with the L-BFGS solver, (b) `class_weight='balanced'` automatically adjusts weights inversely proportional to class frequency, handling the ~47/53% imbalance, and (c) the L2 regularization (`C=1.0`) prevents overfitting on the 50,000-dimensional TF-IDF space.

### Configuration Details

| Component | Parameter | Value | Justification |
|-----------|-----------|-------|---------------|
| **TF-IDF** | `max_features` | 50,000 | Large enough to capture the full vocabulary (memes have diverse lexicon) while limiting memory. Captures ~99% of the observed unigram+bigram vocabulary. |
| | `ngram_range` | (1, 2) | Unigrams capture individual discriminative words; bigrams capture fixed expressions and collocations that are signal-rich for misogyny detection ("make me", "shut up", "stay home"). |
| | `sublinear_tf` | True | Applies `1 + log(tf)` to dampen the influence of very frequent words (articles, pronouns) that carry little discriminative power. |
| **LogReg** | `C` | 1.0 | Standard inverse regularization strength. Values tested: {0.01, 0.1, 1.0, 10.0}; C=1.0 was optimal on validation. |
| | `class_weight` | 'balanced' | Automatically sets `w_class = n_samples / (n_classes × n_samples_class)`, compensating for binary class imbalance. |
| | `max_iter` | 1,000 | Sufficient for L-BFGS convergence on this dataset size. |
| **MultiOutput** | wrapper | `MultiOutputClassifier` | Treats each of the 4 sub-type labels as an independent binary classification problem (binary relevance approach). |

### Cascaded Pipeline (Baseline)

The baseline follows the same cascaded architecture as the transformer model to ensure a fair comparison:

1. **Stage 1 — Binary**: TF-IDF + LR trained on **all 5,250 training samples** → predicts `misogynous` on the 1,500 test samples.
2. **Stage 2 — Multi-label**: `MultiOutputClassifier(LogisticRegression)` trained **only on the ~2,655 misogynous training samples** → predicts 4 sub-type labels.
3. **Cascade logic**: If Stage 1 predicts `misogynous=0` → the 4 multi-label predictions are forced to `[0,0,0,0]`. Only samples predicted as misogynous are forwarded to Stage 2.

**Why cascade the baseline too?** Using the same cascade structure for both baseline and transformer ensures that performance differences are attributable solely to the model's quality, not to architectural differences. If the baseline used flat prediction while the transformer used a cascade, any comparison would conflate the effect of the model with the effect of the pipeline design.

### What does the baseline capture — and what does it miss?

**Captures**:
- Explicit lexical markers of misogyny: slurs, insults, gendered derogatory terms.
- N-gram patterns that are statistically associated with specific labels (e.g., "kitchen" → stereotype, "body" → objectification).

**Misses** (motivating the transformer approach):
- **Context-dependent meaning**: "she looks good" can be neutral or objectifying depending on context. TF-IDF treats these identically because it is a bag-of-words model with no notion of word order or dependence.
- **Negation and sarcasm**: "Women are definitely NOT bad at driving" contains the word "bad" which pushes TF-IDF toward a misogynous prediction, even though the sentence is sarcastic/negating.
- **Implicit misogyny**: Many misogynous memes contain no explicit slurs — the harm comes from implication, which requires understanding context, world knowledge, and pragmatics that linear models cannot learn.

---

## 6. Transformer Model: Selection and Justification

### The Importance of Domain-Specific Pre-training

Transfer learning from pre-trained language models (PLMs) is the dominant paradigm in modern NLP. However, **not all PLMs are created equal** for a given downstream task. The key insight from the literature (Gururangan et al., 2020; "Don't Stop Pretraining") is that the **domain match** between the pre-training corpus and the target data is often more predictive of downstream performance than model size or architecture.

For misogyny detection in memes, the target domain has very specific characteristics:
- **Informal register**: grammar violations, sentence fragments, all-caps emphasis.
- **Internet slang and abbreviations**: "ngl", "smh", "lmao", "bruh".
- **Deliberate misspellings and creative typography**: "b*tch", "w0man".
- **Irony and sarcasm**: literal meaning inverted by context or tone.
- **Hashtag-style markers**: "#NotAllMen", "#KnowYourPlace".

A model pre-trained on Wikipedia and BookCorpus (like standard BERT) has never seen this kind of text during pre-training, and its internal representations are tuned for formal, well-edited prose.

### Chosen Model: `cardiffnlp/twitter-roberta-base-hate`

| Property | Description |
|----------|-------------|
| **Architecture** | RoBERTa-base (12 layers, 768 hidden, 12 attention heads) |
| **Parameters** | ~125M (comparable to BERT-base) |
| **Pre-training Stage 1** | Continued pre-training on ~124M tweets — adapts embeddings and attention patterns to social media language |
| **Pre-training Stage 2** | Fine-tuned on HatEval (hate speech vs. immigrants and women) + OffComEval (offensive language identification) |
| **Tokenizer** | BPE (Byte-Pair Encoding) with 50,265 sub-word tokens, trained on the tweet corpus |
| **Source** | Cardiff NLP group (TweetEval benchmark) |

### Detailed Justification for the Choice

1. **Domain match (most important factor)**: Memes use the same linguistic register as tweets — informal, abbreviated, ironic, and laden with internet culture. The model's embeddings have been continuously pre-trained on 124M tweets, meaning it has already learned representations for slang ("simp", "thot"), abbreviations, emoji references, and the pragmatic patterns of online discourse. This drastically reduces the *distribution shift* between pre-training and inference.

2. **Task-specific transfer from hate speech fine-tuning**: The model was fine-tuned on HatEval (which specifically includes hate speech targeting women, directly relevant to misogyny) and OffComEval (offensive content identification). This means the model's internal representations have already been partially aligned to distinguish hostile/derogatory content from neutral content. Our fine-tuning refines this alignment to the specific MAMI categories rather than learning from scratch.

3. **RoBERTa over BERT**: RoBERTa (Liu et al., 2019) improves on BERT through: (a) dynamic masking (new random masks each epoch vs. BERT's static masks), (b) removal of the Next Sentence Prediction objective (found to be noise), (c) larger batches and more pre-training data. These improvements yield consistently better representations, especially for downstream classification tasks.

4. **Cost-effectiveness**: At 125M parameters, this model is significantly smaller than large alternatives (DeBERTa-v3-large at 304M, Llama at 7B+) while achieving competitive or superior performance on hate speech benchmarks, specifically because of domain match. A 2.5x larger model pre-trained on the wrong domain will underperform a correctly-matched smaller model (as we experimentally verified — see "Discarded models" below).

### Discarded Models — Experimental Evidence

Three alternative models were experimentally evaluated during the project's iterative development:

| Model | Params | Pre-training Corpus | Result | Reason for Rejection |
|-------|--------|---------------------|--------|---------------------|
| `distilbert-base-uncased` | 66M | BookCorpus + Wikipedia | Binary F1 ~0.75, ML ~0.40 | Formal-text domain gap + distillation loses nuance |
| `bert-base-uncased` | 110M | BookCorpus + Wikipedia | Binary F1 ~0.77, ML ~0.42 | Better than DistilBERT but still suffers from domain gap |
| `microsoft/deberta-v3-base` | 184M | Diverse general corpus | Training instability, regression to ~0.74 | Despite larger size, the diverse but non-social-media corpus did not help; DeBERTa's disentangled attention mechanism showed training instability with our FGM adversarial setup |

**Key lesson**: All three models share the same fundamental limitation — they were pre-trained on **formal text** (books, encyclopedias, news articles) and lack exposure to the linguistic distribution of social media. The ~124M tweet pre-training of `twitter-roberta-base-hate` provides a decisive advantage that cannot be compensated by model size alone. DeBERTa-v3, despite being 50% larger and architecturally more sophisticated, performed worse because its representations are optimized for formal text structure.

### Model Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `BERT_MAX_LEN` | 128 | Meme texts are short (median ~15 words, 95th percentile ~60 words). 128 BPE tokens covers >99% of samples without excessive padding. Longer sequences would waste GPU memory and computation on empty padding tokens. |
| `BERT_BATCH` | 32 | Standard batch size for BERT fine-tuning (Devlin et al., 2019 recommend 16 or 32). 32 provides a good gradient estimation per step while fitting within typical VRAM constraints (~4-6 GB for a 125M model at fp16). Combined with `grad_accum=2`, the effective batch size is 64 (§9). |

---

## 7. Multi-Task Learning Architecture (MTL)

### Theoretical Background

Multi-Task Learning (Caruana, 1997) is a training paradigm where a single model is trained simultaneously on multiple related tasks that share a common representation layer. The core hypothesis is that **tasks that share underlying features benefit from being learned together**: the inductive bias from one task acts as a regularizer for the other, and the shared representation must capture information useful for both tasks, leading to more general and robust features.

In our setting, the two tasks are:
- **Binary classification**: Is this meme misogynous? (7,500 samples available)
- **Multi-label classification**: Which sub-types of misogyny are present? (only ~2,655 misogynous samples, but the backbone sees all 7,500 through the binary task)

These tasks are highly related — they share the same semantic space (misogynous content) and the same linguistic patterns. The features that distinguish misogynous from non-misogynous content are directly related to (and overlap with) the features that distinguish shaming from stereotype or objectification from violence.

### The Problem with Separate Models (Previous Approach)

In the project's v2 iteration, **two independent models** were trained:
- **Binary model**: RoBERTa fine-tuned on all 7,500 samples → backbone learns from the full dataset.
- **Multi-label model**: A *separate* RoBERTa fine-tuned **only on ~2,655 misogynous samples** → backbone only sees 35% of the data.

This created three problems:
1. **Data starvation for the ML backbone**: The multi-label model's backbone only sees 2,655 samples, wasting the rich signal from the 4,845 non-misogynous samples. These non-misogynous samples contain valuable *negative* signal — they teach the backbone what misogyny is *not*, which is equally important for discriminative features.
2. **No feature sharing**: Two completely independent backbones (~250M parameters total) learn redundant representations without any cross-task transfer.
3. **Doubled inference cost**: Two separate forward passes at inference time, doubling latency and memory requirements.

### Solution: Multi-Task Model with Shared Backbone

The MTL architecture uses a **single shared RoBERTa backbone** with **two task-specific classification heads**:

```
              Input text
                  │
                  ▼
┌─────────────────────────────────┐
│    Shared RoBERTa Backbone       │
│    (125M params, ALL 7,500       │
│     samples, both gradients)     │
│                                  │
│    [CLS] token → 768-dim vector │
└───────────┬─────────┬────────────┘
            │         │
   ┌────────┴───┐ ┌───┴────────┐
   │ Binary Head│ │  ML Head   │
   │ Drop(0.1)  │ │ Drop(0.1)  │
   │ Dense(768) │ │ Dense(768) │
   │ Tanh       │ │ Tanh       │
   │ Drop(0.1)  │ │ Drop(0.1)  │
   │ Proj(→ 2)  │ │ Proj(→ 4)  │
   └────────────┘ └────────────┘
```

### Implementation Details

```python
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, n_ml_labels=4):
        self.backbone = AutoModel.from_pretrained(model_name)
        h = config.hidden_size    # 768 for RoBERTa-base
        dp = config.hidden_dropout_prob  # 0.1
        # Each head replicates RoBERTa's classification head architecture:
        # dropout → dense(h→h) → tanh activation → dropout → projection
        self.dropout   = nn.Dropout(dp)
        self.bin_dense = nn.Linear(h, h)     # 768 → 768
        self.bin_proj  = nn.Linear(h, 2)     # 768 → 2 (binary classes)
        self.ml_dense  = nn.Linear(h, h)     # 768 → 768
        self.ml_proj   = nn.Linear(h, n_ml_labels)  # 768 → 4 (sub-type logits)

    def forward(self, input_ids, attention_mask, **kwargs):
        # Extract [CLS] token representation from the backbone
        cls = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]  # shape: (batch, 768)

        # Binary head: dropout → dense → tanh → dropout → projection
        b = self.dropout(cls)
        b = torch.tanh(self.bin_dense(b))
        b = self.dropout(b)
        binary_logits = self.bin_proj(b)  # shape: (batch, 2)

        # Multi-label head: same architecture, independent weights
        m = self.dropout(cls)
        m = torch.tanh(self.ml_dense(m))
        m = self.dropout(m)
        ml_logits = self.ml_proj(m)  # shape: (batch, 4)

        return binary_logits, ml_logits
```

**Why replicate RoBERTa's classification head architecture?** The `dense → tanh → dropout → projection` pattern comes from the original RoBERTa paper's classification head. This design provides a non-linear transformation of the `[CLS]` representation before projection, which is more expressive than a simple linear layer. Using `tanh` (vs. ReLU or GELU) matches the original implementation and provides bounded output in [-1, 1], which stabilizes training. Each head has its own independent weights to specialize for its task.

### MTL Advantages — Detailed Analysis

| Advantage | Explanation |
|-----------|-------------|
| **More data for the backbone** | The backbone receives gradient updates from BOTH tasks on every batch. The binary loss updates on all 7,500 samples; the ML loss updates on the ~47% that are misogynous. Net effect: the backbone sees 2.8× more gradient signal than a ML-only model. This is the single biggest contributor to the +2.0% ML F1 improvement over v2. |
| **Implicit regularization** | The binary task acts as an *auxiliary task* that prevents the backbone from overfitting to the multi-label objective. Because the two tasks share the backbone, the representations must be useful for *both*, which constrains the representation space and improves generalization. This is the MTL "regularization effect" described by Ruder (2017). |
| **Richer representations** | Features learned to distinguish misogynous from non-misogynous content are directly useful for the multi-label head. For example, the backbone may learn an "objectification detector" neuron as part of the binary task (since objectification strongly correlates with misogyny), which the ML head can directly leverage. |
| **Efficiency** | A single model (1 forward pass, ~125M params) produces both predictions simultaneously, vs. two separate models (2 forward passes, ~250M params total). This halves inference latency and disk footprint. |
| **Coherent learning** | Both heads share the same representation, ensuring consistency — the features that say "this is misogynous" are the same features available to say "specifically, it's stereotype + objectification." |

### Joint Loss Function — Theoretical Justification

The total loss is a weighted sum of two components:

```
L_total = L_binary(ALL samples) + ml_weight × L_multilabel(ONLY misogynous samples)
```

| Loss | Type | Applied to | Justification |
|------|------|------------|---------------|
| **L_binary** | `CrossEntropyLoss(weight=[1.0, neg/pos])` | All batch samples | Standard cross-entropy for binary classification. Class weights `[1.0, neg_count/pos_count]` ≈ `[1.0, 0.977]` compensate for the moderate binary class imbalance, ensuring equal effective contribution from both classes to the gradient. |
| **L_multilabel** | `BCEWithLogitsLoss(pos_weight=...)` | Only samples where `misogynous=1` | Binary Cross-Entropy with logits (numerically stable sigmoid + BCE in one pass). Applied per-label independently (binary relevance). `pos_weight` per label compensates for severe per-label imbalance within the misogynous subset. |

**Per-label pos_weight values** (computed from training misogynous subset):

| Label | pos_weight | Interpretation |
|-------|-----------|----------------|
| shaming | 2.85 | Most under-represented → positive samples get 2.85× the weight |
| stereotype | 0.65 | Most frequent → positives are slightly *down-weighted* to prevent dominance |
| objectification | 1.19 | Near-balanced → slight positive upweight |
| violence | 3.78 | Rarest label → strongest upweight to prevent the model from ignoring it |

**Why `pos_weight` instead of `class_weight`?** In multi-label setting, each label is an independent binary problem. `pos_weight` in `BCEWithLogitsLoss` scales the loss contribution of positive samples directly: `loss = -[pos_weight × y × log(σ(x)) + (1-y) × log(1-σ(x))]`. This is more targeted than class_weight because it operates at the per-label level.

**Why BCEWithLogitsLoss over Asymmetric Loss (ASL)?** ASL (Ben-Baruch et al., 2020) was experimentally tested in an earlier iteration. It introduces asymmetric margins between positive and negative examples to handle long-tail distributions. However, in our setting ASL caused **gradient collapse**: the hard negative mining mechanism zeroed out gradients for too many samples when combined with the already-filtered misogynous subset, leaving the ML head with insufficient gradient signal to learn. BCE with per-label `pos_weight` is more stable and produced equivalent or superior results.

**ml_weight = 1.0 — Why equal weighting?** The `ml_weight` parameter controls the relative importance of the multi-label loss. We tested values in {0.5, 1.0, 2.0, 5.0}:
- `ml_weight=0.5`: Under-weights ML loss → ML head converges slowly, binary head dominates.
- `ml_weight=1.0`: Balanced — both tasks contribute equally to gradient magnitude.
- `ml_weight=2.0`: Over-weights ML loss → ML head overfits to the smaller misogynous subset.
- `ml_weight=5.0`: ML loss dominates → binary head performance degrades, no net benefit.

The equal weighting `ml_weight=1.0` was empirically optimal on validation.

---

## 8. Adversarial Training: FGM

### Theoretical Background

Adversarial training was introduced by Goodfellow et al. (2015) in the context of adversarial examples — inputs intentionally perturbed to fool a model. The key insight is that training on these adversarial perturbations makes models more robust and improves generalization.

In NLP, we cannot directly "perturb" discrete text tokens (adding 0.01 to a word makes no sense), but we *can* perturb the **continuous embedding vectors** that represent tokens. This is the foundation of adversarial training for NLP: perturb the learned embeddings during training to expose the model to slightly "corrupted" representations, forcing it to be robust to small input variations.

The **Fast Gradient Method (FGM)**, proposed by Miyato et al. (2017, "Adversarial Training Methods for Semi-Supervised Text Classification"), is the simplest and most computationally efficient adversarial training technique. It computes a single perturbation step in the gradient direction.

### FGM Mechanism — Step by Step

At each training step, the following occurs:

1. **Standard forward + backward**: Compute the loss $L$ on the original embeddings $e$ and backpropagate to compute $\nabla_e L$ (the gradient of the loss with respect to the embedding weights).

2. **Compute adversarial perturbation**: The perturbation $\delta$ is computed as:

$$\delta = \epsilon \cdot \frac{\nabla_e L}{\|\nabla_e L\|}$$

This is a unit-norm perturbation in the direction of steepest loss increase, scaled by $\epsilon$. The normalization $\|\nabla_e L\|$ ensures that the perturbation magnitude is controlled regardless of the gradient magnitude.

3. **Attack — apply perturbation**: The embedding weights are temporarily modified: $e' = e + \delta$

4. **Adversarial forward + backward**: A second forward pass is run with the perturbed embeddings $e'$, producing an adversarial loss $L_{adv}$. This loss is backpropagated, accumulating additional gradients.

5. **Restore**: The embedding weights are reverted to their original values: $e' \to e$

6. **Optimizer step**: The optimizer updates parameters using the accumulated gradients from *both* the standard and adversarial passes. This means each parameter receives gradient signal from both "clean" and "corrupted" inputs.

### Implementation

```python
class FGM:
    def __init__(self, model, epsilon=1.0, emb_name='word_embeddings'):
        self.model, self.epsilon, self.emb_name = model, epsilon, emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if self.emb_name in name and param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()  # Save original
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isinf(norm):
                    param.data.add_(self.epsilon * param.grad / norm)  # Perturb

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]  # Revert to original
        self.backup = {}
```

### Configuration and Justification

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `epsilon` | 1.0 | Controls the magnitude of the adversarial perturbation. This value is the standard in the literature for BERT-family models (Miyato et al., 2017; Zhu et al., 2020). Lower values (e.g., 0.1) produce perturbations too small to provide regularization benefit. Higher values (e.g., 5.0) can destabilize training by pushing embeddings too far from the learned manifold, causing loss spikes and convergence failures. ε=1.0 represents the sweet spot, empirically validated across multiple NLP tasks. |
| `emb_name` | `word_embeddings` | FGM perturbs **only** the word embedding weight matrix, not positional embeddings, layer norms, or attention weights. This is the standard choice because: (a) word embeddings are the most directly related to input semantics — perturbing them simulates lexical variation in the input, and (b) perturbing other parameters would change the model's computation in ways unrelated to input variation, adding noise rather than useful regularization. |

### Why FGM is Particularly Effective for Meme Classification

1. **Simulates lexical noise**: Meme texts are inherently noisy — typos, abbreviations, creative spellings, and OCR errors are common. FGM perturbations in embedding space effectively simulate this noise, training the model to be invariant to small lexical variations. A word like "b*tch" vs. "bitch" vs. "biatch" maps to different input tokens but should produce similar predictions; FGM helps achieve this by exposing the model to embedding-space variations during training.

2. **Addresses irony robustness**: Ironic statements often differ from sincere ones by subtle linguistic cues (e.g., "obviously", "clearly", "sure") that can be easily perturbed. FGM forces the model to make robust predictions even when these subtle markers are slightly distorted.

3. **Low computational overhead**: FGM requires exactly one additional forward pass per training step (~50% more time per step). This is amortized when considering the alternative — training for more epochs to achieve similar generalization — and the consistent +1-3% F1 improvement makes it highly cost-effective.

4. **No architectural changes required**: FGM is purely a training-time technique that modifies embedding weights temporarily. The final saved model is identical in architecture to a non-FGM model; the improvement comes from the training dynamics.

### FGM vs. Other Adversarial Methods

| Method | Description | Why not used |
|--------|-------------|-------------|
| **PGD** (Projected Gradient Descent) | Multi-step perturbation with projection back to ε-ball | 3-5× slower (multiple forward passes per step); marginal F1 improvement over FGM for text tasks |
| **FreeLB** | Multi-step with gradient accumulation | More complex implementation; similar performance to FGM on hate speech benchmarks |
| **SMART** | Smoothness-inducing adversarial regularization | Requires a separate regularization loss term; added complexity not justified for our task size |

FGM was selected as the **optimal cost/benefit tradeoff**: single-step perturbation, minimal code complexity, consistent F1 improvement, and negligible impact on training time.

---

## 9. Optimization Strategy

The training loop combines several established best practices for fine-tuning pre-trained language models. Each choice is individually justified below.

### Optimizer: AdamW

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Algorithm** | AdamW (Loshchilov & Hutter, 2019) | AdamW is the **standard optimizer for transformer fine-tuning** because it properly decouples weight decay from the adaptive learning rate. Unlike Adam with L2 regularization, AdamW applies weight decay directly to the parameters (not through the gradient), resulting in more effective regularization. |
| **Learning Rate** | 3e-5 | This is the canonical starting point for BERT/RoBERTa fine-tuning (Devlin et al., 2019 recommend {2e-5, 3e-5, 5e-5}). 3e-5 provides a good balance: large enough for head layers to converge within a few epochs, but small enough (with the 10× decay for the backbone — see below) to avoid catastrophic forgetting of pre-trained representations. |
| **Weight Decay** | 0.01 | Applied to all parameters except biases and LayerNorm weights. The 1% decay prevents weight magnitudes from growing unboundedly during fine-tuning, acting as a regularizer that pushes weights toward zero unless they actively reduce the loss. |
| **Betas** | (0.9, 0.999) | Default AdamW momentum coefficients. β₁=0.9 means the first-moment estimate (gradient mean) has an effective window of ~10 steps; β₂=0.999 means the second-moment estimate (gradient variance) has an effective window of ~1000 steps, providing very stable adaptive learning rate scaling. |

### Layer-wise Learning Rate Decay

```python
backbone params → lr / 10 = 3e-6  (lower layers, pre-trained)
head params     → lr      = 3e-5  (upper layers, randomly initialized)
```

**Justification**: This technique, known as **discriminative fine-tuning** (Howard & Ruder, 2018, ULMFiT), addresses the fundamental asymmetry in fine-tuning: the pre-trained backbone already has high-quality representations learned from 124M tweets, while the classification heads are randomly initialized.

- **Backbone (lr=3e-6)**: A 10× lower learning rate preserves the pre-trained representations while allowing gradual adaptation to the target task. Without this, the early training steps (when head gradients are large and random) would destroy the backbone's pre-trained features — a phenomenon known as **catastrophic forgetting**. The 10× factor is the most common choice in the literature; more aggressive decay (100×) can prevent the backbone from adapting at all.

- **Heads (lr=3e-5)**: The randomly initialized heads need a higher LR to converge quickly. Since their weights start from random values (Xavier initialization via PyTorch defaults), a low LR would require many epochs to reach a useful parameter space.

**Weight decay is also differentiated**:
- **No decay** for: `bias`, `LayerNorm.weight`, `layer_norm.weight` — these parameters control normalization and should not be penalized toward zero, as doing so would disrupt the model's internal normalization scheme.
- **With decay (0.01)** for: all other weight matrices — standard L2-style regularization.

### Schedule: Cosine Annealing with Linear Warmup

```python
warmup_ratio = 0.1  →  first 10% of total_steps: linear warmup from 0 to lr_max
remaining 90%: cosine decay from lr_max to ~0
```

| Phase | Steps | Behavior | Purpose |
|-------|-------|----------|---------|
| **Linear Warmup** | 0 → 10% of total | LR increases linearly from 0 to `lr_max` | Prevents "early training shock": at the start of fine-tuning, the randomly initialized heads produce large, noisy gradients. A low initial LR limits the damage these noisy gradients can cause to the pre-trained backbone. By the end of warmup, the heads have begun to converge, producing more meaningful gradients. |
| **Cosine Decay** | 10% → 100% of total | LR follows $lr_t = lr_{max} \cdot \frac{1}{2}(1 + \cos(\pi \cdot t / T))$ | Smooth, gradual decay allows the model to make fine-grained adjustments in later epochs. Unlike step decay (which causes loss spikes at each step), cosine decay is continuous and produces smoother training curves. The final LR approaches 0, effectively "annealing" the model into a local minimum. |

**Why cosine over linear or step decay?** Cosine scheduling (Loshchilov & Hutter, 2017, SGDR) has been empirically shown to produce better generalization than linear or step-function schedules across NLP benchmarks. The gradual slowdown in later epochs allows the model to exploit the local loss landscape without overshooting.

### Gradient Accumulation

```python
grad_accum = 2  →  effective batch size = BERT_BATCH × grad_accum = 32 × 2 = 64
```

**Justification**: The effective batch size is a critical hyperparameter for transformer fine-tuning:

1. **Larger effective batch = more stable gradients**: Each gradient update averages over 64 samples instead of 32. This is especially important for the multi-label loss, which is only computed on the misogynous subset of each batch (~47% of samples). With batch_size=32, only ~15 samples contribute to the ML gradient per step — very noisy. With effective batch_size=64, ~30 samples contribute — significantly more stable.

2. **No additional VRAM required**: Unlike increasing `BERT_BATCH` from 32 to 64 (which would double VRAM usage), gradient accumulation achieves the same effective batch size by accumulating gradients over 2 micro-batches before the optimizer step. Each micro-batch still uses only 32-sample VRAM.

3. **Trade-off**: Gradient accumulation halves the number of optimizer steps per epoch, which slightly slows convergence in wall-clock time per step savings. This is why we use `grad_accum=2` (not 4 or 8) — it doubles effective batch size while keeping the total update count reasonable.

### Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Justification**: Gradient clipping is a safety mechanism that prevents **gradient explosion**, which can cause NaN losses and training divergence. It is particularly important in our setup for three reasons:

1. **Adversarial gradient accumulation**: Each training step accumulates gradients from two forward passes (standard + FGM adversarial). This effectively doubles the gradient magnitude, increasing the risk of explosion.
2. **Mixed precision (fp16)**: AMP introduces numerical sensitivity at low precision; clipping prevents occasional large-magnitude fp16 gradients from destabilizing the model.
3. **Multi-task gradients**: Gradients from the binary and multi-label losses can occasionally conflict (pointing in opposite directions for shared backbone parameters), and the resulting gradient vector can have unusually large norms.

The `max_norm=1.0` threshold is the standard for BERT-family models (used in the original BERT, RoBERTa, and DeBERTa fine-tuning recipes).

### Mixed Precision Training (AMP fp16)

```python
_USE_AMP = DEVICE.type == 'cuda'
scaler = torch.cuda.amp.GradScaler(enabled=_USE_AMP)
```

**What AMP does**: Automatic Mixed Precision (Micikevicius et al., 2018) runs the forward pass and backward pass in **fp16** (16-bit floating point) instead of fp32, while keeping master weights and certain critical operations (like loss computation and softmax) in fp32.

**Why it matters**:
1. **Speed**: fp16 operations on modern NVIDIA GPUs use Tensor Cores, providing ~2-3× throughput improvement over fp32.
2. **Memory**: fp16 activations and gradients use half the VRAM, allowing larger batch sizes or longer sequences.
3. **Numerical safety**: The `GradScaler` handles the fp16 stability problem: it scales the loss by a large factor before backward (to prevent gradient underflow in fp16), then unscales before the optimizer step. If gradients overflow (become inf), the scaler skips that step and reduces the scale factor.

**Automatic fallback**: When no CUDA GPU is available (`DEVICE.type != 'cuda'`), AMP is automatically disabled (`_USE_AMP = False`), and all computation runs in fp32 on CPU. This ensures the notebook is portable across different hardware configurations.

### Early Stopping

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `patience` | 8 epochs | Allows sufficient time for the cosine schedule to provide late-training improvements. Too low (e.g., 3) would stop training before the warm-up benefit is fully realized. Too high (e.g., 15) would waste computation on overfitting epochs. 8 is a commonly used value for BERT fine-tuning setups. |
| **Monitored metric** | `0.4 × bin_f1 + 0.6 × ml_f1` (combined F1) | Optimizes for both tasks simultaneously. The 0.6 weight for ML reflects two factors: (a) it is the harder task (higher room for improvement), and (b) it is the task that benefits most from multi-task training. |
| `epochs` (max) | 30 | Upper bound; rarely reached due to early stopping. |
| **Best model checkpoint** | CPU clone of state_dict when combined F1 improves | Saves the best model weights to CPU memory at each improvement, ensuring the final model corresponds to the best validation performance, not the last epoch. |

**Combined metric justification (detailed)**: Why not just monitor ML F1?
- Monitoring only ML F1 could cause the binary head to overfit (since its performance is not checked).
- Monitoring only binary F1 would stop training before the ML head has fully converged.
- The combined metric `0.4×bin + 0.6×ml` ensures both heads are performing well, with a slight emphasis on ML (the harder and more informative task).
- Validation is performed in **cascaded mode** (identical to final evaluation), so the metric measures end-to-end pipeline performance, including error propagation from the binary head to the ML head.

---

## 10. Cascaded Pipeline: End-to-End Evaluation

### Design Rationale

The cascaded pipeline is the central architectural decision of the classifier. Rather than predicting all labels in a flat multi-label setup (5 outputs: misogynous + 4 sub-types), we implement a **two-stage cascade** where the binary prediction gates the multi-label prediction.

This design is motivated by the **logical dependency** between the labels: the sub-type labels (`shaming`, `stereotype`, `objectification`, `violence`) are only semantically meaningful when `misogynous = 1`. A non-misogynous meme should never have any active sub-type label — this is a hard constraint, not a statistical tendency. The cascade enforces this constraint architecturally.

### Cascade Architecture

```
Input text
    │
    ▼
┌──────────────────────────┐
│   Binary Head (MTL)      │
│   argmax(logits) → 0/1   │
│                          │
│   Confidence: softmax    │
│   provides probability   │
└──────────┬───────────────┘
           │
     ┌─────┴──────┐
     │             │
  pred = 0      pred = 1
     │             │
     ▼             ▼
  [0,0,0,0]    ┌───────────────────────┐
  (no labels)  │   ML Head (MTL)       │
  → forced     │   sigmoid(logits)     │
    zeros      │   per-label threshold │
               │   comparison          │
               └───────────────────────┘
                       │
                       ▼
              [shaming, stereo, obj, viol]
              (each 0 or 1, independent)
```

### Cascade Advantages — Detailed Analysis

1. **Eliminates logical inconsistencies**: Without the cascade, a flat model might predict {misogynous=0, shaming=1} — an incoherent output that would require post-processing to resolve. The cascade makes such outputs *structurally impossible*.

2. **Reduces multi-label false positives**: In a flat model, the multi-label heads see all samples and can produce false positive sub-type predictions for non-misogynous memes. In the cascade, if the (more reliable) binary head correctly says "not misogynous", all sub-type labels are automatically zero, preventing spurious activations.

3. **Realistic end-to-end evaluation**: The cascaded metrics reflect what would happen in a production deployment — binary errors propagate to the multi-label stage. A binary false negative (misogynous meme missed) causes 4 automatic multi-label false negatives (all sub-types missed). A binary false positive causes a non-misogynous meme to be forwarded to the ML head, potentially generating sub-type false positives. This error propagation is captured in the end-to-end F1 metric, providing a realistic assessment of system performance.

4. **Single forward pass efficiency**: Despite the logical two-stage design, the implementation runs a **single forward pass** through the MTL model, which produces both binary and multi-label logits simultaneously. The "cascade" is implemented purely in the prediction logic (masking ML outputs to zero when binary prediction is 0), not as two sequential model calls. This means there is no additional latency compared to a flat multi-label model.

### Implementation

```python
@torch.no_grad()
def cascaded_predict(model, texts, thresholds):
    """
    Full cascaded pipeline using the multi-task model.
    One forward pass → binary head decides misogynous/not →
    multi-label head predictions kept only for predicted-misogynous.
    """
    model.eval()
    ds     = BertDataset(texts)
    loader = DataLoader(ds, batch_size=BERT_BATCH, shuffle=False, num_workers=0)

    all_bin, all_probs = [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=_USE_AMP):
            bin_logits, ml_logits = model(**batch)
        all_bin.extend(bin_logits.argmax(1).cpu().tolist())
        all_probs.append(torch.sigmoid(ml_logits).cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    ml_preds = []
    for i, bin_pred in enumerate(all_bin):
        if bin_pred == 1:
            ml_preds.append([int(probs[i, k] > thresholds[k]) for k in range(len(thresholds))])
        else:
            ml_preds.append([0, 0, 0, 0])  # Cascade: force zeros

    return all_bin, ml_preds
```

**Design notes**:
- `torch.no_grad()` disables gradient computation, reducing memory by ~50% during inference.
- `model.eval()` disables dropout, ensuring deterministic predictions.
- `autocast(enabled=_USE_AMP)` uses fp16 for inference when available, providing ~2x speedup.
- The threshold comparison `probs[i, k] > thresholds[k]` uses per-label optimized thresholds (§11) instead of a flat 0.5.

### Error Propagation Analysis

The cascaded design introduces a specific failure mode: **binary errors multiply into multi-label errors**. Understanding this is critical for interpreting the results:

| Binary Error Type | Effect on Multi-label | Frequency |
|-------------------|-----------------------|-----------|
| **False Negative** (misogynous → pred 0) | All 4 sub-type labels are forced to [0,0,0,0], causing up to 4 false negatives. This is the most damaging error type. | ~9% of test misogynous samples |
| **False Positive** (not misog. → pred 1) | The non-misogynous text is forwarded to the ML head, which may produce false positive sub-type labels. | ~8% of test non-misogynous samples |
| **True Positive/True Negative** | Correct cascading; ML head operates on the correct subset. | ~83% of test samples |

The binary head's ~17.5% error rate (262/1,500) is the primary bottleneck for overall system performance. Improving binary accuracy would have a disproportionate effect on the end-to-end multi-label F1.

---

## 11. Per-Label Threshold Tuning

### The Problem with a Uniform 0.5 Threshold

After applying `sigmoid()` to the multi-label logits, the standard approach is to predict a label as positive when the probability exceeds 0.5. This assumes that the model's probability outputs are **well-calibrated** — that a predicted probability of 0.5 truly represents a 50/50 chance. In practice, this assumption almost never holds for fine-tuned transformer models, for several reasons:

1. **Class imbalance**: When a label has very few positives (e.g., violence at ~21%), the model learns to predict low probabilities even for true positives, because predicting "negative" is right 79% of the time. A 0.5 threshold is too high and misses many true positives.

2. **Label frequency bias**: Conversely, for the most frequent label (stereotype at ~61%), the model may assign high probabilities even to borderline cases, because the base rate is high. A lower threshold might be optimal to capture the full recall.

3. **Model calibration drift**: Fine-tuning on a small dataset changes the model's output distribution. The sigmoid probabilities from a fine-tuned model do not have the same calibration properties as the pre-trained model's outputs.

### Solution: Per-Label Grid Search

Instead of using a uniform threshold, we perform an **independent grid search** for each of the 4 labels, finding the threshold that maximizes F1-score on the validation set:

```python
def tune_ml_thresholds(model, texts, labels, n_steps=80):
    """Per-label threshold search on misogynous validation subset."""
    probs = model.predict_probabilities(texts)  # shape: (n_val_mis, 4)
    best_thresholds = []
    for label_idx in range(4):
        best_threshold, best_f1 = 0.5, 0.0
        for threshold in np.linspace(0.1, 0.9, n_steps):  # 80 candidate thresholds
            f1 = f1_score(y[:, label_idx], (probs[:, label_idx] > threshold).astype(int))
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
        best_thresholds.append(best_threshold)
    return best_thresholds
```

### Search Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Search range** | [0.1, 0.9] | Broad enough to capture extreme cases (low for frequent labels, high for rare labels). We exclude [0, 0.1) and (0.9, 1] because thresholds near extreme values would produce predictions that are either all-positive or all-negative. |
| **Steps** | 80 | Provides a granularity of ~0.01 across the [0.1, 0.9] range, which is sufficiently fine-grained. Tested with 200 steps and results were identical (plateau regions in the F1 curve are wider than 0.01). |
| **Data** | Validation set, **misogynous-only samples** (~379 samples) | Critical design choice — see justification below. |
| **Metric** | F1-score per label (binary) | F1 balances precision and recall for each label independently. We maximize F1 rather than accuracy because accuracy is dominated by the majority class. |

### Why Misogynous-Only Samples for Threshold Tuning?

This is one of the most important methodological decisions in the pipeline. In the cascade (§10), the multi-label head is **only invoked for samples predicted as misogynous** by the binary head. This means:

- At inference time, the ML head only sees inputs that the binary head classified as misogynous.
- The distribution of these inputs is different from the full dataset: it contains all true misogynous samples (that were correctly identified) plus some non-misogynous false positives.
- Tuning thresholds on the full validation set (including non-misogynous samples with `[0,0,0,0]`) would introduce a massive number of easy true negatives that artificially inflate all metrics and bias thresholds toward high values (because the model correctly assigns low probabilities to non-misogynous samples, making high thresholds appear optimal).

By tuning on the misogynous-only validation subset, we optimize thresholds for the exact distribution the ML head faces at inference time.

### Results and Interpretation

| Label | Optimal Threshold | Val F1 | Interpretation |
|-------|-------------------|--------|----------------|
| **shaming** | 0.465 | 0.5918 | Close to default 0.5. Shaming has moderate prevalence (~27%) and the model's probability distribution is relatively well-calibrated for this label. |
| **stereotype** | 0.100 | 0.7675 | Extremely low threshold — the most aggressive. Stereotype is the most frequent label (~61%) and the model tends to under-predict it (conservative probabilities). Lowering the threshold to 0.1 maximizes recall, capturing many borderline cases that would be missed at 0.5. The high val F1 (0.7675) confirms this is the correct direction. |
| **objectification** | 0.292 | 0.6812 | Low-medium threshold. Objectification is the second most frequent label (~46%) and benefits from a lower threshold to boost recall. |
| **violence** | 0.586 | 0.5818 | Above default 0.5. Violence is the rarest label (~21%) and the model occasionally over-predicts it (assigns moderate probabilities to non-violent misogynous content). A higher threshold increases precision, reducing false positives for this sensitive category. |

**Key insight**: The thresholds span a wide range (0.100 to 0.586), confirming that a uniform 0.5 would be substantially suboptimal. The per-label search recovers +2-5% F1 per label compared to the uniform threshold, and this improvement propagates to the macro-averaged E2E metric.

### Persistence

Optimized thresholds are saved to `ml_thresholds.json` for:
- **Reproducibility**: Exact thresholds can be loaded for consistent evaluation across runs.
- **Production use**: The Streamlit demo and ONNX inference load thresholds from this file, ensuring the same decision boundaries are used everywhere.

---

## 12. Error Analysis

### Purpose and Methodology

Error analysis goes beyond aggregate metrics (F1, precision, recall) to understand *why* the model fails. By categorizing errors into interpretable patterns, we can: (a) identify systematic biases, (b) prioritize future improvements, and (c) provide users with actionable understanding of the model's limitations.

Our approach uses **automatic error tagging** with regex heuristics applied to misclassified test samples. Each incorrect prediction is tagged with one or more failure categories based on properties of the input text:

### Failure Categories — Definition and Rationale

| Tag | Heuristic | Description | Why it causes errors |
|-----|-----------|-------------|---------------------|
| `SHORT_TEXT` | `len(words) < 8` | Very short texts with insufficient context | Short texts provide too few semantic cues for the transformer to form confident predictions. The [CLS] representation from <8 tokens is under-determined. |
| `NEGATION` | `\b(not\|never\|no\|n't\|without\|barely\|hardly\|neither\|nor)\b` | Presence of negation | Negation inverts meaning ("women are NOT objects" vs. "women are objects"). While transformers have *some* ability to handle negation, it remains one of the hardest linguistic phenomena for NLU systems. The model may focus on the content words (`women`, `objects`) and miss the negation cue. |
| `IRONY/SARCASM` | `\b(obviously\|clearly\|apparently\|sure\|just\|totally\|definitely\|righ+t)\b` | Irony/sarcasm markers | Sarcasm inverts the literal meaning of the entire utterance. "Oh sure, women are GREAT at driving" is sarcastic and misogynous, but the literal surface meaning is positive. Detecting sarcasm from text alone (without vocal intonation or visual cues) is an open research problem even for state-of-the-art models. |
| `AMBIGUOUS_HEDGE` | `\b(maybe\|perhaps\|kind of\|sort of\|almost\|quite\|rather\|could be)\b` | Hedging/uncertainty language | Hedges weaken the force of a statement, making the misogynous intent ambiguous: "Maybe women should just stay home" vs. "Women should stay home." The model must assess whether the hedge negates the harmful intent or simply softens it. |
| `OTHER` | Default (no pattern matches) | Errors without an obvious linguistic pattern | The largest category — these are errors due to deep semantic ambiguity, subtle/implicit misogyny, subjective labeling in the dataset, or content that requires visual context (the meme image) to interpret correctly. |

**Important clarification**: `OTHER` is **not a prediction class** — it is a diagnostic category for the error analysis. It indicates that the failure cannot be attributed to a simple linguistic pattern (negation, irony, etc.) and likely requires deeper semantic understanding or multimodal information to resolve.

### Error Extraction Process

The top 20 worst errors are extracted for both tasks:

- **Binary errors**: All samples where `pred ≠ true`, tagged with failure categories and showing text snippets.
- **Multi-label errors**: All samples with at least one incorrect sub-type label, sorted by `n_err` (number of incorrect labels per sample, 1-4), showing true vs. predicted label sets.

This prioritization (worst errors first) ensures that the analysis focuses on the most informative failure cases — the ones that are most wrong and most illuminating about systemic problems.

### Observed Error Patterns (from last run)

| Error Type | Binary Count | Binary % | Multi-label Count | ML % |
|------------|-------------|----------|-------------------|------|
| `OTHER` | 180 | 68.7% | 475 | 66.0% |
| `SHORT_TEXT` | 38 | 14.5% | 101 | 14.0% |
| `NEGATION` | 28 | 10.7% | 95 | 13.2% |
| `IRONY/SARCASM` | 16 | 6.1% | 62 | 8.6% |
| `AMBIGUOUS_HEDGE` | 3 | 1.1% | 12 | 1.7% |
| **Total errors** | **262 / 1,500** | **17.5%** | **720 / 1,500** | **48.0%** |

### Key Observations

1. **`OTHER` dominates both tasks** (~67%): The majority of errors have no simple linguistic explanation. This strongly suggests that the errors are caused by: (a) **implicit misogyny** — harmful content that requires understanding cultural context, power dynamics, and pragmatic implicature, beyond what lexical patterns can capture; (b) **subjective labeling** — annotator disagreement in the original MAMI dataset, where the boundary between misogynous and non-misogynous is genuinely ambiguous; (c) **visual-dependent meaning** — memes where the text alone is neutral or ambiguous, and the misogynous meaning comes from the combination with the image.

2. **`SHORT_TEXT` is proportionally more damaging for multi-label** (14.0% vs. 14.5%): Short texts have even less signal for distinguishing between sub-types than for the binary task. Predicting "misogynous" from a few words is feasible if they contain strong markers ("b*tch"), but distinguishing *which type* of misogyny from 5-7 words is extremely difficult.

3. **`IRONY/SARCASM` is disproportionately high in multi-label** (8.6% vs. 6.1%): Sarcasm is especially damaging for sub-type classification because it can invert or activate multiple labels simultaneously. A sarcastic meme about cooking ("Oh sure, because a woman's place is DEFINITELY in the kitchen") should be tagged as `stereotype`, but the sarcastic framing makes it appear non-misogynous to the model.

4. **Multi-label errors (720) are 2.75× more frequent than binary errors (262)**: This gap reflects: (a) the inherent difficulty of fine-grained classification vs. binary detection, (b) error propagation from the cascade (each binary error causes up to 4 ML errors), and (c) the smaller training set for the ML head (2,655 vs. 7,500 samples).

---

## 13. RAG Pipeline: Meme Generation

### Purpose and Motivation

The RAG (Retrieval-Augmented Generation) pipeline serves a dual purpose in this project:

1. **Synthetic data generation**: It produces new meme captions that can be used to augment the evaluation dataset, testing the classifier's ability to generalize to LLM-generated content (which has different distributional properties than human-written memes).

2. **Demonstrating an end-to-end NLP system**: Beyond classification, the RAG pipeline showcases the complete NLP engineering stack — document ingestion, chunking, vector storage, hybrid retrieval, and generation — all orchestrated through a REST API. This meets the project requirement of demonstrating both analysis (classification) and generation (meme creation) capabilities.

### RAG Architecture

The pipeline follows the standard RAG design pattern (Lewis et al., 2020): instead of generating text purely from the LLM's parametric knowledge, the system **retrieves relevant documents** from a knowledge base and includes them in the generation prompt as context.

```
┌──────────┐  PDF/DOCX   ┌───────────┐  store    ┌─────────┐
│ Scripts  │ ──────────→ │ Flask API │ ───────→ │  MinIO  │
│          │             │   :5000   │          │  :9000  │
│          │  question   │           │  embed   │         │
│Streamlit │ ──────────→ │ parse →   │ ───────→ ┌─────────┐
│  :8501   │             │ chunk →   │          │ChromaDB │
└──────────┘             │ retrieve →│ ←─────── │  :8000  │
                         │ generate  │          └─────────┘
                         │           │ prompt   ┌─────────┐
                         │           │ ───────→ │ llama   │
                         │           │ ←─────── │  :8080  │
                         └───────────┘          └─────────┘
```

### Components — Detailed Explanation

#### 13.1 Ingestion (`ingestion.py`)

The ingestion module handles converting uploaded documents (PDF, DOCX) into raw text that can be chunked and embedded:

- **PDF parsing** via `pdfplumber`: Extracts text page-by-page, preserving paragraph structure. `pdfplumber` was chosen over `PyPDF2` because it handles complex layouts (multi-column, tables) more reliably.
- **DOCX parsing** via `python-docx`: Extracts paragraph text from Word documents.
- **Validation**: The module rejects scanned PDFs (image-only pages with no extractable text) and empty DOCX files, returning informative error messages rather than silently processing garbage input.
- **Raw document storage**: Uploaded files are also stored in MinIO (S3-compatible object storage) for traceability and reprocessing.

#### 13.2 Chunking (`chunking.py`)

Chunking splits long documents into smaller segments that can be independently embedded and retrieved. The chunk size directly affects retrieval quality: too large → chunks contain diluted information; too small → chunks lack sufficient context.

Three strategies are implemented, offering a trade-off between simplicity and semantic quality:

| Strategy | Description | Parameters | When to use |
|----------|-------------|------------|-------------|
| **Fixed-Size** | Sliding window of N words with overlap. | `CHUNK_SIZE=256`, `CHUNK_OVERLAP=32` (12.5% overlap) | Default. Fast, predictable chunk sizes, works well for homogeneous documents. The overlap ensures that information spanning a chunk boundary is not lost. |
| **Recursive** | Hierarchical splitting: first by paragraphs, then by sentences, then by fixed size. | Preserves natural author boundaries (paragraph breaks > sentence breaks > word breaks). | Better for structured documents with clear sections, where paragraph boundaries carry semantic information. |
| **Semantic** | Computes cosine similarity between consecutive sentences using a SentenceTransformer. Cuts where similarity drops below a threshold (0.5). | Requires `all-MiniLM-L6-v2` model for sentence embeddings. | Best semantic coherence per chunk, but computationally expensive and sensitive to the threshold parameter. |

**Default choice justification**: Fixed-size chunking with `CHUNK_SIZE=256` and `CHUNK_OVERLAP=32` provides a good balance. The overlap of 32 words (~12.5%) ensures that queries matching text near a chunk boundary will find the relevant chunk in at least one of its overlapping versions. While semantic chunking produces better-quality chunks, the additional computation (one embedding pass per sentence) is not justified for our use case, where the documents are relatively homogeneous (meme descriptions and hate speech definitions).

#### 13.3 Retrieval (`retrieval.py`)

The retrieval module implements three retrieval methods, each with different strengths:

| Method | Implementation | Strength | Weakness |
|--------|---------------|----------|----------|
| **Dense** | ChromaDB + `all-MiniLM-L6-v2` embeddings + cosine similarity | Captures **semantic similarity** — can match paraphrases and conceptual equivalents even with zero lexical overlap. E.g., "body shame" matches "criticize appearance." | Can fail with rare technical terms or proper nouns that are not well-represented in the embedding space. |
| **BM25** | `rank_bm25.BM25Okapi` with local corpus persisted in pickle | Captures **exact lexical matching** — excels at finding documents containing specific terms, abbreviations, or unusual words. | Cannot handle paraphrases: "women in the kitchen" will NOT match "female workers should stay home" despite identical meaning. |
| **Hybrid** (default) | **Reciprocal Rank Fusion (RRF)** of Dense + BM25 rankings | Combines both strengths, producing a ranking that benefits from semantic understanding AND lexical precision. | Slightly more complex implementation; two retrieval passes per query. |

**Reciprocal Rank Fusion (RRF)** combines the rankings from both methods without needing to calibrate their raw scores:

$$\text{score}_{\text{RRF}}(\text{chunk}) = \sum_{i \in \{\text{dense, BM25}\}} \frac{1}{k + \text{rank}_i(\text{chunk}) + 1}$$

where $k = 60$ is the smoothing constant (standard value from Cormack et al., 2009).

**Why RRF over score-level fusion?** Dense retrieval and BM25 produce scores on completely different scales (cosine similarity ∈ [-1, 1] vs. BM25 scores ∈ [0, ∞)). Normalizing and summing these scores would require careful calibration that varies per query and corpus. RRF operates on **ranks** (ordinal positions), which are inherently comparable — rank 1 means "best match" regardless of the underlying scoring mechanism.

#### 13.4 Generation (`generation.py`)

- **LLM**: **Mistral 7B Instruct v0.2** (GGUF quantized, Q4_K_M, ~4.4GB on disk). Mistral 7B was chosen because: (a) it is the smallest instruction-tuned model that produces coherent, styled text, (b) the Q4_K_M quantization reduces memory from ~14GB (fp16) to ~4.4GB while maintaining >95% of the model's quality, and (c) it supports the Mistral Instruct prompt format natively.
- **Serving**: via **llama.cpp** server with GPU offloading (`-ngl 35` → 35 layers on GPU). llama.cpp provides a lightweight HTTP API compatible with OpenAI-style requests, without needing a heavy framework like vLLM or TGI.
- **Prompt format**: Mistral Instruct wrapping: `[INST] system prompt + context + user query [/INST]`.
- **Context window**: 4096 tokens (`-c 4096`) — sufficient for RAG context (typically 1-2K tokens of retrieved chunks) plus the prompt and response.
- **Generation parameters**:
  - `temperature=0.85`: Slightly above the default (0.7) to encourage diversity in generated memes. Higher temperatures produce more creative but less coherent output; 0.85 balances creativity with coherence.
  - `top_p=0.95`: Nucleus sampling — considers only the tokens whose cumulative probability exceeds 95%, filtering out the long tail of improbable tokens.
  - `repeat_penalty=1.2`: Penalizes token repetition, preventing the model from generating repetitive phrases (a common failure mode in small LLMs).

#### 13.5 ReAct Agent (`agent.py`) — Bonus

The agent module implements the **ReAct** (Reasoning + Acting) pattern (Yao et al., 2023), enabling iterative multi-step retrieval:

1. The LLM receives a question and reasons about what information it needs (`Thought: I need to find examples of body shaming in memes...`).
2. The LLM chooses an action: `SEARCH[query]` to retrieve documents, or `ANSWER[text]` to produce a final response.
3. If `SEARCH` is chosen → the query is sent to the hybrid retrieval system → results are injected back as `Observation`.
4. The loop continues for a maximum of **5 iterations**, allowing the model to refine its searches based on what it finds.

**Justification**: Simple single-pass RAG can fail when a question requires information from multiple perspectives or when the initial retrieval doesn't return relevant documents. The ReAct loop allows the model to "think" about what it found, identify gaps, and issue follow-up searches. This is especially useful for complex queries like "compare different types of misogyny in memes" where a single keyword search would be insufficient.

#### 13.6 Meme Generation (`generate_memes_rag.py`)

The generation script produces a balanced synthetic evaluation dataset:

- **50 memes** total: 10 per category (shaming, stereotype, objectification, violence) + 10 neutral.
- **Generation process per meme**:
  1. Select a category and craft a category-specific retrieval query (e.g., "examples of body shaming memes" for the shaming category).
  2. Retrieve context from the RAG knowledge base using hybrid retrieval.
  3. Build a prompt with: (a) system instruction specifying the desired category, (b) retrieved context as examples, (c) explicit instruction to generate one meme caption.
  4. Call the LLM via the llama.cpp API.
  5. Filter responses that look like model refusals ("I cannot generate harmful content...") and retry with a rephrased prompt.
- **Output**: `generated_memes_rag.csv` in TSV format with the same column structure as the original MAMI dataset, enabling direct use with the classifier's evaluation pipeline.

#### 13.7 Generation Examples

Below are sample meme captions produced by the RAG pipeline for each category, demonstrating that the generated content is type-distinguishable and suitable for classifier evaluation:

| Category | Generated Caption |
|----------|-------------------|
| **Neutral** | "Why don't scientists trust atoms? / Because they make up everything!" |
| **Neutral** | "When your cat discovers the joy of playing with water bottles... and your productivity level drops to zero" |
| **Shaming** | "Women, why can't you even lift a box of chocolate without asking for help?!" |
| **Shaming** | "Women, why can't you balance a glass of water and walk in heels at the same time?" |
| **Stereotype** | "Why did the woman go to IKEA alone? To assemble her emotions" |
| **Stereotype** | "Why cook when you can UberEats and cry about it on Pinterest?" |
| **Objectification** | "Women, who needs a microwave when you've got that toaster oven body?" |
| **Objectification** | "Why change the light bulb when she can just adjust her makeup to match the new brightness?" |
| **Violence** | "When she doesn't clean up after dinner: 'I'll just have to rearrange some of your organs instead'" |
| **Violence** | "If she doesn't comply with my demands, I'll just add her name to my Restraining Order list" |

**Analysis of generated content quality**:
- The **neutral** samples are clearly benign humor with no gendered targeting — confirming the model can produce non-harmful output when instructed.
- **Shaming** samples target personal capabilities and competence, attacking women's ability to perform everyday tasks.
- **Stereotype** samples reinforce traditional gender roles (cooking, shopping, emotional instability) — the hallmark patterns of this sub-type.
- **Objectification** samples reduce women to physical/utilitarian attributes, using metaphors that compare women to household objects.
- **Violence** samples contain explicit or implicit threats of physical harm — the most extreme category.

This type-distinguishability validates that the RAG pipeline produces semantically coherent content that meaningfully tests the classifier's ability to differentiate sub-types. The higher binary F1 (0.8824) and lower ML F1 (0.4291) on these generated memes vs. the original test set (see §18) provides an interesting distribution-shift analysis.

---

## 14. Infrastructure and Deployment

### Design Philosophy

The infrastructure follows the **microservices pattern**: each component (storage, vector DB, LLM server, API, frontend) runs as an independent Docker container with a well-defined API contract. This design provides: (a) **isolation** — a crash in the LLM server doesn't affect document storage, (b) **scalability** — individual services can be scaled independently, (c) **reproducibility** — the entire system is defined in a single `docker-compose.yml` that any developer can launch with one command.

### Docker Compose — 5 Services

| Service | Image | Port | Purpose | Resource Requirements |
|---------|-------|------|---------|-----------------------|
| **MinIO** | `minio/minio:latest` | 9000 (API), 9001 (Console) | S3-compatible object storage for raw uploaded documents. Provides durable, versioned storage with a web console for management. | Low (disk-bound) |
| **ChromaDB** | `chromadb/chroma:0.6.3` | 8000 | Vector database storing chunk embeddings. Supports cosine similarity search natively. Chosen over Pinecone/Weaviate for simplicity and local deployment (no cloud dependency). | Medium (RAM for indices) |
| **llama.cpp** | `ghcr.io/ggml-org/llama.cpp:server-cuda` | 8080 | LLM inference server running Mistral 7B (GGUF). Uses CUDA for GPU acceleration (`-ngl 35` offloads 35 of ~40 layers to GPU). Provides an OpenAI-compatible HTTP API for text generation. | High (GPU: ~4GB VRAM) |
| **Flask API** | Custom Dockerfile | 5000 | REST orchestration layer that coordinates the full RAG pipeline: receives queries, calls ChromaDB for retrieval, calls llama.cpp for generation, returns formatted responses. Also handles document upload, parsing, and chunking. | Low-Medium (CPU-bound) |
| **Streamlit** | Custom Dockerfile | 8501 | Web UI for interactive RAG interaction. Provides file upload, query interface, and response display. Communicates exclusively with Flask API (never directly with other services). | Low |

### Health Checks and Dependency Management

Docker Compose health checks ensure that services start in the correct order and that dependent services only begin when their dependencies are fully operational:

| Service | Health Check | Interval | Justification |
|---------|-------------|----------|---------------|
| **MinIO** | `GET /minio/health/live` | 10s | Standard MinIO liveness endpoint. Quick to respond even during heavy I/O. |
| **ChromaDB** | `GET /api/v1/heartbeat` | 10s | ChromaDB's official heartbeat. Returns 200 only when the vector index is ready to serve queries. |
| **llama.cpp** | `GET /health` | 30s, start_period: 300s | Model loading takes 60-120s (loading 4.4GB GGUF + GPU memory allocation). The 300s start period prevents premature health check failures during this loading phase. |
| **Flask API** | Depends on: MinIO (healthy) + ChromaDB (healthy) + llama.cpp (healthy) | — | Only starts when all three backends are operational, preventing connection errors on first request. |
| **Streamlit** | Depends on: Flask API (healthy) | — | Only starts when the orchestration layer is ready. |

This cascading dependency chain ensures a clean startup: MinIO + ChromaDB + llama.cpp (parallel) → Flask API → Streamlit.

### Environment Variables (Centralized Configuration)

All RAG pipeline configuration is managed through environment variables, centralized in `config.py`. This follows the **12-factor app** methodology (configuration in the environment, not in code):

| Variable | Default Value | Description | Why configurable |
|----------|---------------|-------------|------------------|
| `MINIO_ENDPOINT` | `minio:9000` | MinIO endpoint | Allows switching to external S3 in production |
| `CHROMA_HOST` | `chromadb` | ChromaDB host | Docker service name resolution; override for external deployment |
| `LLM_URL` | `http://llama:8080` | llama.cpp server URL | Allows pointing to a different LLM server (e.g., a remote GPU instance) |
| `CHUNK_SIZE` | 256 | Chunk size in words | Tune for different document types |
| `CHUNK_OVERLAP` | 32 | Overlap between consecutive chunks | Balance redundancy vs. boundary information loss |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for embeddings | Upgrade to larger/better models without code changes |
| `RETRIEVAL_METHOD` | `hybrid` | Retrieval strategy (dense/bm25/hybrid) | Experiment with different strategies per use case |
| `TOP_K` | 5 | Number of chunks to retrieve per query | Control context length vs. precision trade-off |
| `MAX_UPLOAD_MB` | 50 | Maximum document upload size | Security: prevent memory exhaustion from oversized uploads |

### Persistent Volumes

```yaml
volumes:
  minio_data:    # Raw documents — survives container restarts
  chroma_data:   # Vector indices — avoids re-embedding on restart
  flask_data:    # BM25 corpus pickle + document registry metadata
```

**Why persistent volumes?** Without them, every `docker-compose down` / `docker-compose up` cycle would lose all uploaded documents, vector embeddings, and BM25 indices — requiring complete re-ingestion. Persistent volumes ensure that data survives container lifecycle events.

---

## 15. Streamlit Demo

### Purpose

The Streamlit demo provides an interactive web interface for real-time meme classification, allowing users to test the model on arbitrary text inputs without writing code. It serves both as a demonstration tool (for stakeholders and reviewers) and as a debugging/exploration tool (for developers to test edge cases).

### Functionality (`src/app.py`)

The demo implements the complete classification pipeline in a user-friendly interface:

1. **Input**: Free text area where users can type any meme caption, or selection from predefined examples that illustrate different classification scenarios.
2. **Cascaded inference**: The input text is processed through the full pipeline — tokenization → binary head → cascade gate → multi-label head with optimized thresholds.
3. **Output**: Visual metric cards showing:
   - Binary result: misogynous/not misogynous with confidence probability.
   - Per sub-type results: each of the 4 categories with icon, probability, and status (detected/not detected).
4. **Real-time processing**: Inference runs on the loaded PyTorch model (or ONNX model if configured), with sub-second response times (~1ms per sample).

### Predefined Examples — Designed to Test Edge Cases

| Example | Expected Classification | Why included |
|---------|------------------------|-------------|
| "Women are underrepresented in technology leadership roles." | Non-misogynous | Tests that factual statements about gender inequality are NOT classified as misogynous — distinguishes social commentary from hate speech. |
| "Women shouldn't worry about STEM, they just feel better with people." | Stereotype | Tests detection of implicit stereotyping that doesn't use explicit slurs but reinforces the "women are emotional, not logical" trope. |
| "She's only good for looking at, not thinking." | Shaming + Objectification | Tests multi-label detection — this sentence simultaneously shames intellectual capability and objectifies. |
| "Girls like that deserve what's coming to them." | Violence | Tests implicit violence detection — no explicit threat, but "deserve what's coming" implies impending harm. |
| "Oh sure, women are SO bad at driving. Totally not a myth." | Adversarial (sarcastic) | Tests the model's weakest area — sarcasm. The literal words contain a misogynous statement, but the sarcastic framing ("Oh sure", "Totally not a myth") negates it. This is the hardest case and often triggers a false positive. |

These examples were specifically chosen to probe the model's strengths and known failure modes, providing users with an immediate understanding of what the model can and cannot handle.

---

## 16. ONNX Export

### What is ONNX and Why Export?

ONNX (Open Neural Network Exchange) is an open standard format for representing machine learning models. Exporting to ONNX serves three production-critical purposes:

1. **Portability**: The model can be loaded and run by any ONNX-compatible runtime — Python (`onnxruntime`), C++ (`onnxruntime-cpp`), JavaScript (`onnxruntime-web`), or mobile (ONNX Runtime Mobile). This eliminates the PyTorch dependency in production, reducing container size from ~2GB (with PyTorch + CUDA) to ~200MB (ONNX Runtime only).

2. **Performance optimization**: ONNX Runtime automatically applies graph-level optimizations that PyTorch typically does not:
   - **Operator fusion**: Combines sequential operations (e.g., MatMul + Add + GELU → FusedMatMulAddGELU) into single custom kernels.
   - **Constant folding**: Pre-computes operations on constant tensors at load time rather than at every inference call.
   - **Memory planning**: Optimizes tensor memory allocation to minimize peak memory usage.
   - These optimizations typically provide 10-30% inference speedup over PyTorch eager mode.

3. **Production deployment**: In a production service (e.g., behind a REST API or in a mobile app), using ONNX Runtime avoids the overhead and complexity of a full PyTorch installation. It also provides more predictable latency characteristics.

### Implementation Details

The export requires a thin wrapper class because `torch.onnx.export` expects a module whose `forward` method takes only tensor arguments (not keyword arguments like `input_ids=...`):

```python
class _OnnxWrapper(nn.Module):
    """Wrapper exposing (input_ids, attention_mask) → (binary_logits, ml_logits)"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

torch.onnx.export(
    wrapper,
    (dummy_ids, dummy_mask),         # Example tensors for tracing
    path,
    input_names=['input_ids', 'attention_mask'],
    output_names=['binary_logits', 'ml_logits'],  # BOTH heads in one graph
    dynamic_axes={                   # Allow variable batch size
        'input_ids':      {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
        'binary_logits':  {0: 'batch_size'},
        'ml_logits':      {0: 'batch_size'},
    },
    opset_version=14,                # Minimum opset supporting all RoBERTa operations
)
```

**Key design decisions**:
- **Dual output in a single graph**: Unlike separate models, the ONNX graph contains both heads in one computation graph. This maintains the MTL efficiency — a single forward pass produces both binary and multi-label predictions.
- **Dynamic batch size**: The `dynamic_axes` configuration allows the ONNX model to accept any batch size at inference time, not just the dummy input size. This is essential for production use where batch sizes vary.
- **Opset version 14**: The minimum ONNX opset version that supports all operations used by RoBERTa (specifically, LayerNormalization with the parameters used by the model). A warning may appear about version conversion, but the export succeeds at opset 18 with full compatibility.

### Saved Artifacts

| File | Description | Size |
|------|-------------|------|
| `multitask_model_weights.pt` | PyTorch state_dict of the complete MTL model (backbone + both heads) | ~480MB |
| `ml_thresholds.json` | Optimized per-label thresholds `[0.465, 0.100, 0.292, 0.586]` | <1KB |
| `multitask_model.onnx` | ONNX computation graph with both heads, dynamic batch size | ~480MB |

All three files are stored in `src/models/trained/` and are versioned together to ensure consistency between weights, thresholds, and the ONNX graph.

---

## 17. Code Quality and Tests

### Static Analysis

| Tool | Result | Purpose |
|------|--------|---------|
| **black** | Consistent formatting (line-length 100) | Automatic code formatting — eliminates style debates and ensures consistent readability across all modules. Line length of 100 (vs. default 88) accommodates the longer lines common in ML code (model configurations, tensor operations). |
| **pylint** | Score: 10.00/10 | Static code analysis for Python best practices: unused imports, naming conventions, missing docstrings, unreachable code. A perfect score confirms compliance with PEP 8 and pylint's additional checks. |
| **pytest** | 61 unit tests, all passing | Automated test suite covering all RAG pipeline components. |

### Test Coverage

The 61 unit tests cover the following functional areas, ensuring that each component works correctly in isolation:

| Test Area | What is tested | Why it matters |
|-----------|---------------|----------------|
| **Document parsing** (PDF, DOCX) | Correct text extraction, rejection of scanned/empty files, handling of malformed inputs | Prevents garbage-in-garbage-out: if parsing fails silently, all downstream components produce meaningless results |
| **Chunking strategies** | Fixed-size, recursive, and semantic chunking produce correct chunk sizes, overlaps, and boundaries | Incorrect chunking (e.g., no overlap, wrong size) would degrade retrieval quality |
| **Retrieval** (dense, BM25, hybrid) | Correct ranking behavior, RRF fusion logic, handling of empty corpora | Retrieval is the core of RAG — incorrect retrieval = irrelevant context = poor generation |
| **Flask API endpoints** | Correct HTTP response codes, request validation, error handling for malformed requests | Ensures the API contract is reliable for the Streamlit frontend and external consumers |
| **LLM generation** | Prompt formatting, response parsing, refusal detection | Ensures the generation module correctly interfaces with llama.cpp and handles edge cases |

---

## 18. Results Summary

### Comparison Table — Test Set (1,500 samples)

| Task | TF-IDF + LR | MTL + FGM | Gain | Relative Improvement |
|------|-------------|-----------|------|---------------------|
| **Binary (F1-Macro)** | 0.7879 | **0.8253** | **+0.0374** | +4.7% |
| **Multi-label E2E (F1-Macro cascaded)** | 0.4829 | **0.5199** | **+0.0370** | +7.7% |

### Per-Class Results — Binary (MTL + FGM)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| non-misogynous | 0.81 | 0.85 | 0.83 | 742 |
| misogynous | 0.84 | 0.81 | 0.82 | 758 |
| **macro avg** | **0.83** | **0.83** | **0.83** | **1,500** |

**Binary results analysis**:
- **Balanced performance**: Both classes achieve comparable F1 (~0.82-0.83), indicating that the weighted cross-entropy loss effectively compensated for the ~47/53% class imbalance. Neither class is sacrificed for the other.
- **Non-misogynous has higher recall (0.85)**: The model is slightly better at identifying non-misogynous content than misogynous content. This is expected because non-misogynous memes tend to have clearer lexical signals (neutral humor, factual statements) than misogynous memes (which can be implicit, ironic, or context-dependent).
- **Misogynous has higher precision (0.84)**: When the model predicts misogynous, it is more often correct. This is desirable in a content moderation context: fewer false accusations of misogyny.

### Per-Class Results — Multi-label E2E Cascaded (MTL + FGM)

| Label | Precision | Recall | F1-Score | Support | Analysis |
|-------|-----------|--------|----------|---------|----------|
| shaming | 0.36 | 0.58 | 0.45 | 206 | Low precision: many false positives. Shaming overlaps linguistically with other categories, especially stereotype. High recall indicates the model catches true shaming cases but also flags many non-shaming texts. |
| stereotype | 0.52 | 0.83 | 0.64 | 458 | Best F1 (0.64) — the most frequent label benefits from the most training signal. Very high recall (0.83) is boosted by the aggressive threshold of 0.100. Precision is moderate because the low threshold also captures borderline cases. |
| objectification | 0.48 | 0.68 | 0.56 | 346 | Balanced performance. Objectification has relatively distinct lexical markers (body-related terms, appearance language) that help the model discriminate. |
| violence | 0.41 | 0.45 | 0.43 | 163 | Lowest F1 — the rarest label (163 test samples) provides the least training signal. Violence is also often implicit ("deserve what's coming") rather than explicit, making it harder to detect from text alone. |
| **macro avg** | **0.44** | **0.64** | **0.52** | **1,173** | Macro-averaged F1 of 0.52 treats all labels equally, meaning the poor performance on violence and shaming pulls down the average. |

**Multi-label results analysis**:
- **Recall > Precision across all labels**: The per-label threshold tuning (§11) optimized for F1, which in our label-imbalanced setting favors recall. This is appropriate for a detection system: it is better to flag suspicious content for human review (high recall) than to miss harmful content (high precision but low recall).
- **Performance correlates with label frequency**: stereotype (most frequent, F1=0.64) > objectification (second, F1=0.56) > shaming (third, F1=0.45) > violence (rarest, F1=0.43). This is a classic long-tail effect: rarer categories receive fewer gradient updates during training.
- **Label confusion matrix insight**: The most common confusion is between *shaming* and *stereotype* — both involve demeaning comments about women's capabilities or roles, but through different mechanisms (personal attack vs. generalized role assignment). The boundary between them is often subjective.

### Results on RAG-Generated Memes (50 samples)

| Task | F1-Macro | Comparison to Test Set |
|------|----------|----------------------|
| **Binary** | **0.8824** | +0.0571 vs. test (0.8253) |
| **Multi-label E2E** | 0.4291 | -0.0908 vs. test (0.5199) |

**Analysis of the divergence**:

1. **Why higher binary F1 on generated memes (0.8824 vs. 0.8253)?** LLM-generated meme captions are more *explicit* than real memes. The Mistral 7B model, even with careful prompting, produces more stereotypical and linguistically clear examples of misogyny (e.g., explicit gender-based insults) compared to real memes, which rely heavily on visual context, cultural references, and implicit meaning. The classifier easily identifies these explicit patterns, leading to higher binary accuracy.

2. **Why lower multi-label F1 on generated memes (0.4291 vs. 0.5199)?** Two factors contribute:
   - **Distribution shift**: Each generated meme was prompted for a single category (e.g., "generate a shaming meme"), producing "pure" single-label examples. The real MAMI dataset has frequent multi-label co-occurrence (e.g., a meme can be both `stereotype` and `objectification`). The classifier was trained on the real distribution and struggles with the artificial single-label distribution of the generated data.
   - **Small sample size sensitivity**: With only 50 memes (10 per category), F1-macro is extremely sensitive to individual errors. A single misclassification in a 10-sample category swings the per-label F1 by ~0.1, causing high variance in the macro average.

### Training Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Trainable parameters | 125,831,430 | ~125M params, dominated by the RoBERTa backbone (~124M). The two classification heads add only ~1.8M params (4 × Linear(768, 768) + 4 × Linear(768, 2/4)). |
| Epochs executed | 23 / 30 (early stop) | Early stopping triggered at epoch 23 with patience 8, meaning the best model was found at epoch 15. The 8 additional epochs confirmed that no further improvement was possible, validating the patience parameter. |
| Training time | 902.6s (~15.0 min) | ~39s per epoch on CUDA GPU. This includes both the standard and adversarial forward passes (FGM), AMP fp16 computation, and gradient accumulation. Without FGM, training would take ~600s (~26s/epoch), confirming the ~50% overhead from FGM. |
| Best combined val (0.4×bin + 0.6×ml) | 0.6375 | This combined metric weights ML more heavily (0.6) because it is the harder task. The value 0.6375 = 0.4×0.8160 + 0.6×0.5214, showing that binary validation is strong while ML validation has room for improvement. |
| Best bin val F1 | 0.8160 | Validation F1 is slightly lower than test F1 (0.8253), which is normal — the validation set (750 samples) has higher variance than the test set (1,500 samples). |
| Best ml val F1 | 0.5214 | Validation ML F1 is close to test ML F1 (0.5199), confirming that the model does not overfit to the validation set and generalizes well. |
| Inference speed | 1.058 ms/sample | Average inference time per sample on GPU with AMP fp16. This includes tokenization, forward pass, and cascade logic. Fast enough for real-time applications (>900 samples/second). |

### Optimized Per-Label Thresholds

| Label | Threshold | Val F1 | Shift from 0.5 | Interpretation |
|-------|-----------|--------|-----------------|----------------|
| shaming | 0.465 | 0.5918 | -0.035 | Minimal shift — shaming is moderately represented and well-calibrated |
| stereotype | 0.100 | 0.7675 | -0.400 | Massive downshift — the model is very conservative about predicting stereotype unless forced to be more aggressive |
| objectification | 0.292 | 0.6812 | -0.208 | Significant downshift to boost recall on this under-predicted label |
| violence | 0.586 | 0.5818 | +0.086 | Upshift — raises precision for the rarest label, preventing false positives |

### Historical Project Evolution

| Iteration | Binary F1 | ML E2E F1 | Key Change | Lessons Learned |
|-----------|-----------|-----------|------------|-----------------|
| v1 (DistilBERT separate) | ~0.75 | ~0.40 | Smaller model, formal pre-training corpus, separate models | Domain match matters more than model size |
| v2 (twitter-roberta separate) | 0.8213 | 0.4999 | Domain-matched model, but still separate binary + ML models | ML model limited by small training set (2,655 samples) |
| **v3 (MTL + FGM)** | **0.8253** | **0.5199** | **Shared backbone + adversarial training** | **MTL gives the ML head access to all 7,500 samples via shared representations; FGM adds robustness** |

**Progression analysis**:
- v1 → v2: The largest jump (+0.07 binary, +0.10 ML) came from switching to a domain-matched model. This confirms that **pre-training domain** is the single most important factor for downstream performance.
- v2 → v3: The MTL upgrade provided a smaller but meaningful improvement (+0.004 binary, +0.020 ML). The binary gain is marginal because the binary head already had access to all 7,500 samples in v2. The ML gain (+0.020) is the key MTL benefit: the shared backbone now transfers knowledge from all 7,500 samples to the multi-label task.

### Detailed Results Analysis and Justification

#### Why does MTL+FGM outperform the baseline and separate models?

**1. Binary F1: 0.8253 (+0.0374 over TF-IDF, +0.004 over v2)**

- The +3.7% improvement over TF-IDF is fundamentally due to transformers' ability to capture **contextual dependencies** that bag-of-ngrams models cannot represent. Consider the phrase "good for": in TF-IDF, this bigram has a single weight regardless of context. But "she's good for nothing" (misogynous) and "this recipe is good for dinner" (neutral) require completely different interpretations. The transformer's 12 self-attention layers capture these context-dependent meanings through bidirectional attention over the full input sequence.

- The marginal improvement over v2 (+0.004) is expected: the binary head already had access to 7,500 samples in v2, so the additional MTL benefit is the **implicit regularization** from the auxiliary multi-label task. This forces the backbone to learn representations that are useful for *both* tasks, slightly improving generalization.

- The **262 binary errors** (17.5% error rate) are concentrated in patterns identified by the error analysis (§12), dominated by `OTHER` (68.7%) — implicit misogyny that requires cultural/visual context beyond text.

**2. Multi-label E2E: 0.5199 (+0.037 over TF-IDF, +0.020 over v2)**

- The +2.0% improvement over v2 is the **most significant MTL contribution**. In v2, the ML backbone only saw 2,655 samples. With MTL, the same backbone receives gradient updates from all 7,500 samples (through the binary loss), learning richer and more generalizable representations that directly benefit the multi-label head. This is the core theoretical advantage of MTL (§7) realized in practice.

- FGM adversarial training contributes especially here: embedding perturbations simulate the lexical variations typical of memes (typos, abbreviations, creative spellings), improving the model's robustness to noisy text. The ML task benefits more from FGM than the binary task because the sub-type decision boundaries are finer-grained and more sensitive to lexical variation.

- The **720 multi-label errors** (48.0% error rate) reflect the inherent difficulty of fine-grained sub-type classification:
  - `OTHER` (475): Most errors have no obvious linguistic pattern — they are due to **semantic ambiguity** between sub-types (*is it stereotype or shaming?*), subjective dataset labeling, and binary error propagation.
  - `SHORT_TEXT` (101): Short texts have insufficient signal to distinguish sub-categories.
  - `NEGATION` (95): Negations confuse per-label classification by inverting apparent meaning.
  - `IRONY/SARCASM` (62): Sarcasm is especially damaging in multi-label because it can activate or deactivate multiple labels simultaneously — a sarcastic stereotype comment may or may not actually be stereotyping.

**3. Why does ML E2E remain below 0.55? Structural performance ceiling**

The performance is limited by factors that **cannot be solved by a better text model**:

| Factor | Impact | Potential solution |
|--------|--------|--------------------|
| **Error propagation** | ~17.5% binary error rate × 4 labels = up to 70% of binary errors become 4× ML errors | Improve binary accuracy (e.g., ensemble methods) |
| **Labeling subjectivity** | Boundaries between stereotype/shaming and objectification/stereotype are genuinely ambiguous for human annotators | Use soft labels (probability distributions) instead of hard 0/1 |
| **Text-only limitation** | Many memes derive meaning from the combination of image + text; OCR text alone is insufficient | Add visual features (CLIP, ViLT) for multimodal classification |
| **Dataset size** | ~2,655 misogynous training samples across 4 categories (with multi-label overlap) provides limited examples per category | Data augmentation (back-translation, LLM paraphrasing, generated memes from RAG) |
| **Label co-occurrence** | Binary relevance approach treats each label independently, ignoring correlations (e.g., violence rarely appears without shaming) | Label-correlation models (classifier chains, label embedding) |

**4. RAG Memes: Excellent binary (0.8824) vs. moderate ML (0.4291)**

This divergence is informative about the nature of LLM-generated vs. human-written content:
- **Binary F1 = 0.8824** (higher than real memes): Generated memes are more lexically explicit — the LLM uses stereotypical language patterns that are easy for the classifier to detect. Real memes are subtler, relying on cultural references and visual humor.
- **ML E2E F1 = 0.4291** (lower than real memes): The generated memes follow a "one category per meme" distribution (by design of the generation prompt), while the real dataset has multi-label co-occurrence. The classifier was trained on the co-occurrence distribution and is miscalibrated for the single-label distribution.

---

## 19. Conclusions and Future Work

### Main Contributions

This project delivers five key technical contributions:

1. **Multi-Task Learning Architecture**: A single RoBERTa backbone jointly trained on binary classification (7,500 samples) and multi-label classification (~2,655 misogynous samples), leveraging shared representations so that the multi-label head benefits from all 7,500 samples through the binary gradient path. This is the core architectural innovation, delivering +2.0% ML F1 over the separate-model approach.

2. **FGM Adversarial Training**: Fast Gradient Method adversarial perturbation applied to word embeddings during every training step, providing a consistent +1-3% F1 improvement as a "free" regularizer. Particularly effective for the noisy, informal text domain of meme captions.

3. **Complete RAG Pipeline**: An end-to-end Retrieval-Augmented Generation system using Mistral 7B (locally quantized), ChromaDB vector store, and hybrid retrieval (Dense + BM25 + RRF). Capable of generating type-distinguishable meme captions across all 5 categories (4 misogyny sub-types + neutral), serving both as a data augmentation tool and as a demonstration of generative NLP capabilities.

4. **ReAct Agent**: An iterative agentic pipeline following the ReAct (Reasoning + Acting) pattern, enabling multi-step retrieval for complex queries where a single retrieval pass is insufficient.

5. **Production-Ready Infrastructure**: Docker Compose orchestration of 5 microservices with health checks and dependency management, ONNX model export for framework-independent deployment, interactive Streamlit demo for real-time classification, and a full test suite (61 unit tests, pylint 10/10).

### Error Distribution — Summary

| Error Type | Binary (262 total) | Multi-label (720 total) | Interpretation |
|------------|--------------------|-----------------------|----------------|
| `OTHER` (no pattern) | 180 (68.7%) | 475 (66.0%) | Implicit misogyny, subjective labeling, visual-dependent meaning |
| `SHORT_TEXT` (<8 words) | 38 (14.5%) | 101 (14.0%) | Insufficient textual signal for classification |
| `NEGATION` | 28 (10.7%) | 95 (13.2%) | Meaning inversion confuses the model |
| `IRONY/SARCASM` | 16 (6.1%) | 62 (8.6%) | Literal vs. intended meaning mismatch |
| `AMBIGUOUS_HEDGE` | 3 (1.1%) | 12 (1.7%) | Softened language obscures misogynous intent |

The dominance of `OTHER` (~67% of all errors) across both tasks is the most important finding. It indicates that the remaining errors are **not due to simple linguistic patterns** that could be solved with better feature engineering or more sophisticated text processing. Instead, they reflect **deep semantic ambiguity** that would require:
- Multimodal information (image + text) to resolve visual-dependent memes.
- A larger, more consistently annotated dataset to reduce labeling noise.
- Cultural and contextual knowledge that goes beyond individual text analysis.

### Known Limitations

| Limitation | Impact | Severity |
|------------|--------|----------|
| **Text-only approach** | Loses all visual information from the meme. Many memes are harmless as text but misogynous when combined with the image (and vice versa). | High — likely accounts for a significant portion of `OTHER` errors. |
| **Implicit misogyny detection** | Ironic, sarcastic, and "coded" misogyny (using dog-whistles or euphemisms) is systematically missed. Irony/sarcasm accounts for ~6-9% of errors but is likely under-counted (some `OTHER` errors are undetected sarcasm). | Medium — improving sarcasm detection would substantially reduce errors in both tasks. |
| **Dataset size constraint** | 7,500 total samples (2,655 misogynous) limits the model's ability to learn fine-grained sub-type boundaries, especially for rare categories (violence: only ~480 training samples). | Medium — more data would improve the long-tail categories (violence, shaming). |
| **Error propagation in cascade** | The ~17.5% binary error rate cascades into 4× multi-label errors. A binary false negative silently causes 4 multi-label false negatives. | Medium — fundamental trade-off of the cascade design; mitigated only by improving binary accuracy. |
| **Label co-occurrence modeling** | The binary relevance approach treats each label independently, ignoring label correlations (e.g., violence almost always co-occurs with shaming). | Low — could be addressed with classifier chains or graph neural networks on the label space. |

### Future Work

The following directions address the identified limitations in order of expected impact:

1. **Multimodal classification** (addresses: text-only limitation, `OTHER` errors):
   - Incorporate CLIP (Contrastive Language-Image Pre-training) or ViLT (Vision-and-Language Transformer) to jointly encode the meme image and text.
   - The visual modality would provide crucial context for memes where text alone is ambiguous (e.g., neutral text overlaid on a misogynous image).
   - Expected impact: potentially large improvement, as visual context is key to interpreting many memes.

2. **Data augmentation** (addresses: dataset size, rare label performance):
   - **Back-translation**: Translate meme captions to another language and back (EN → FR → EN) to generate paraphrased training examples.
   - **LLM paraphrasing**: Use the RAG pipeline to generate additional training samples, especially for under-represented categories (violence, shaming).
   - **Synonym replacement**: WordNet-based augmentation for controlled lexical variation.
   - Expected impact: moderate, especially for the long-tail categories.

3. **Sarcasm-aware classification** (addresses: irony/sarcasm errors):
   - Incorporate a sarcasm detection module (pre-trained sarcasm detector) as an additional feature or attention gate.
   - Or use an ensemble: combine the main classifier with a sarcasm-specialized model.
   - Expected impact: moderate — sarcasm accounts for 6-9% of errors, but improving sarcasm handling would also reduce some `OTHER` errors.

4. **Label correlation modeling** (addresses: label co-occurrence):
   - Replace binary relevance with **classifier chains** that model label dependencies sequentially.
   - Or use **label embedding** approaches where labels are embedded in a shared space and predicted jointly.
   - Expected impact: low-moderate — the binary relevance approach is already competitive, and label correlations in MAMI are not extremely strong.

5. **Continual learning loop** (addresses: evolving meme landscape):
   - Use the RAG pipeline to generate new training examples → retrain the classifier → deploy → collect feedback → repeat.
   - This addresses the fundamental challenge that internet memes evolve rapidly, and a static model will degrade over time as new slang, formats, and cultural references emerge.
   - Expected impact: high for long-term deployment, low for static evaluation.

---

*Report generated as part of the capstone project development.*
