# MMCultureQA 2027: Multilingual Multimodal Cultural Question Answering 🖼️🗣️

[![SemEval 2027](https://img.shields.io/badge/SemEval-2027%20Shared%20Task-blue.svg)](https://mmcultureqa-semeval27.github.io/)
[![Dataset: OASIS](https://img.shields.io/badge/Dataset-QCRI%2FMMCQA--SemEval27-orange.svg)](https://huggingface.co/datasets/QCRI/MMCQA-SemEval27)
[![Evaluation: BERTScore](https://img.shields.io/badge/Official%20Metric-BERTScore%20F1-green.svg)](https://mmcultureqa-semeval27.github.io/tasks/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)

> **Working prototype and starter kit for SemEval 2027 MMCultureQA**, featuring multilingual speech recognition (ASR), vision-language reasoning with Cultural Retrieval-Augmented Generation (Cultural RAG), offline reproducible baselines, official BERTScore evaluation, and CodaBench packaging.

---

## Table of Contents

1. [Task Overview](#task-overview)
2. [Why Cultural Grounding?](#why-cultural-grounding)
3. [System Architecture](#system-architecture)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Official Sample Dataset](#official-sample-dataset)
7. [Running the Prototype via CLI](#running-the-prototype-via-cli)
   - [1. Inspect Dataset Splits](#1-inspect-dataset-splits)
   - [2. Run Single-Instance Inference (Task 1 & Task 2)](#2-run-single-instance-inference)
   - [3. Run Batch Predictions](#3-run-batch-predictions)
   - [4. Benchmark & Evaluate](#4-benchmark--evaluate)
   - [5. Package for CodaBench](#5-package-for-codabench)
8. [Interactive Streamlit Web Dashboard](#interactive-streamlit-web-dashboard)
9. [Evaluation Suite & Metrics](#evaluation-suite--metrics)
10. [Cultural Knowledge Base & RAG](#cultural-knowledge-base--rag)
11. [Checkpoints & Rollback History](#checkpoints--rollback-history)
12. [Citation](#citation)

---

## Task Overview

[MMCultureQA](https://mmcultureqa-semeval27.github.io/) is an official shared task featured in **SemEval 2027**, addressing the challenge of **multimodal question answering grounded in local cultural contexts**. Built upon the **OASIS** dataset framework, systems receive an image and a question and must output a short, open-ended textual answer.

The competition comprises two distinct tasks across **18 planned language tracks**:

| Task | Input Modality | Target Output | Challenge Focus |
|---|---|---|---|
| **Task 1: Spoken Visual QA (SQA)** | Image (`.jpg`) + Spoken Audio Question (`.wav`) | Short Open-Ended Text Answer | End-to-end understanding from speech directly |
| **Task 2: Textual Visual QA (QA)** | Image (`.jpg`) + Written Text Question | Short Open-Ended Text Answer | Isolates multimodal & cultural reasoning |

### 18 Planned Language Tracks
- **Arabic Varieties**: Modern Standard Arabic (`msa`), Egyptian Arabic (`arz`), Levantine Arabic (`ajp`).
- **English**: Global English (`en`).
- **South Asian Languages**: Hindi (`hi`), Urdu (`ur`), Bangla (`bn`), Assamese (`as`), Gujarati (`gu`), Marathi (`mr`).
- **European Languages**: Italian (`it`), Spanish (`es`), Portuguese (`pt`).
- **Turkic Languages**: Turkish (`tr`).
- **Horn of Africa Languages**: Amharic (`am`), Oromo (`om`), Somali (`so`), Tigrinya (`ti`).

---

## Why Cultural Grounding?

Standard vision models fail when answering cultural questions because **visual recognition alone is insufficient**. 

For example, when shown an image of interlocking gypsum-like discs:
- A generic computer vision model detects: *"curved architectural discs in a desert setting"*.
- The cultural reference answer requires: *"The building is the **National Museum of Qatar**, and its design is inspired by **desert rose formations**."*

Similarly, questions concerning traditional clothing (*Galabeya*, *Bisht*, *Dishdasha*), ceremonial dances (*Dabke*, *Tanoura*), storytelling (*Hakawati*), hospitality vessels (*Dallah* for *Gahwa*), or Ramadan lanterns (*Fanous*) demand **external cultural knowledge** that is never explicitly written inside the image pixels.

---

## System Architecture

```
                                 ┌────────────────────────────────┐
                                 │   SemEval 2027 MMCultureQA     │
                                 └──────────────┬─────────────────┘
                                                │
                      ┌─────────────────────────┴────────────────────────┐
                      │                                                  │
            [Task 1: Spoken QA]                                [Task 2: Text QA]
            Image + Audio (.wav)                               Image + Text Question
                      │                                                  │
                      ▼                                                  │
         ┌─────────────────────────┐                                     │
         │   Multilingual ASR      │                                     │
         │ (English, Arabic, etc.) │                                     │
         └────────────┬────────────┘                                     │
                      │ (Transcribed Question)                           │
                      └─────────────────────────┬────────────────────────┘
                                                ▼
                                 ┌─────────────────────────────┐
                                 │ Cultural Knowledge Base     │
                                 │ (Regional traditions, food, │
                                 │  landmarks, clothing, etc.) │
                                 └──────────────┬──────────────┘
                                                │ (Retrieved Cultural Context)
                                                ▼
                                 ┌─────────────────────────────┐
                                 │ Multimodal Cultural Reasoner│
                                 │ • Fast Local Baseline       │
                                 │ • Cultural VLM (RAG)        │
                                 └──────────────┬──────────────┘
                                                │ (Open-ended Short Answer)
                                                ▼
                                 ┌─────────────────────────────┐
                                 │ Evaluation & CodaBench      │
                                 │ • Official BERTScore F1     │
                                 │ • SacreBLEU & ROUGE-1/2/L   │
                                 │ • CodaBench Zip Generator   │
                                 └─────────────────────────────┘
```

---

## Project Structure

```
MMCultureQA/
├── app.py                          # Streamlit interactive web dashboard
├── requirements.txt                # Production and prototype dependencies
├── .gitignore                      # Git exclusion rules
├── README.md                       # Comprehensive documentation
├── data/                           # Official OASIS dataset assets
│   ├── images/                     # Real JPEG cultural images
│   ├── audio/                      # Spoken WAV questions (en/, msa/)
│   ├── qa/                         # Textual QA JSONL splits (train, dev)
│   └── sqa/                        # Spoken QA JSONL splits (train, dev)
├── mmcultureqa/                    # Core Python package
│   ├── __init__.py                 # Package exports and version metadata
│   ├── __main__.py                 # CLI entry point (python -m mmcultureqa)
│   ├── config.py                   # Paths, tracks, and parameter configuration
│   ├── dataset.py                  # MMCultureQADataset & record validation
│   ├── asr.py                      # Multilingual ASR transcriber & cache
│   ├── cultural_knowledge.py       # Knowledge base & Cultural RAG retriever
│   ├── cli.py                      # Rich command-line interface
│   ├── models/
│   │   ├── __init__.py             # Solver exports
│   │   ├── base.py                 # Abstract BaseSolver interface
│   │   ├── local_baseline.py       # Zero-GPU offline cultural solver
│   │   ├── cultural_vlm.py         # Multimodal VLM (OpenAI / open models)
│   │   └── pipeline.py             # Unified Task 1 & Task 2 orchestrator
│   └── evaluation/
│       ├── __init__.py             # Evaluation exports
│       ├── metrics.py              # BERTScore F1, SacreBLEU, ROUGE suite
│       ├── evaluator.py            # Granular reporting by language/country
│       └── codabench.py            # CodaBench packager & schema validator
└── tests/                          # Automated unit test suite
    ├── test_dataset.py             # Schema & media validation tests
    ├── test_asr.py                 # Audio transcription & caching tests
    ├── test_cultural_knowledge.py  # Cultural entity retrieval tests
    ├── test_models.py              # Solver & pipeline tests
    ├── test_evaluation.py          # Metric calculation & alignment tests
    └── test_codabench.py           # Submission packaging tests
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/SauhardyaCode/MMCultureQA.git
cd MMCultureQA
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Tests
Verify your installation by running the full test suite:
```bash
python -m unittest discover tests
```

---

## Official Sample Dataset

The repository comes pre-packaged with the official initial sample release from [QCRI/MMCQA-SemEval27](https://huggingface.co/datasets/QCRI/MMCQA-SemEval27).

To refresh or verify the dataset at any time:
```bash
python -m mmcultureqa download
```

### Dataset Schema
Each entry in `data/qa/{train,dev}_{lang}.jsonl`:
```json
{
  "id": "bc22d5a245df0f65c673be9d0c974b903fc080cb02d74f7b166f89a95fa6fbba",
  "image": "images/bc22d5a245df0f65c673be9d0c974b903fc080cb02d74f7b166f89a95fa6fbba.jpg",
  "country": "Egypt",
  "category": "Religion & Spirituality",
  "subcategory": "Holy Texts & Scriptures",
  "question": "What can you infer about the cultural significance of the clothing worn by the seated individuals in the image?",
  "answer": "The colorful and intricately designed clothing suggests a connection to traditional cultural practices, possibly indicating their involvement in a performance, storytelling, or a ceremonial activity in the public square."
}
```

For **Task 1 (Spoken QA)** in `data/sqa/{train,dev}_{lang}.jsonl`, the textual `"question"` field is replaced with `"audio"` pointing to the corresponding spoken WAV file:
```json
{
  "id": "bc22d5a245df0f65c673be9d0c974b903fc080cb02d74f7b166f89a95fa6fbba",
  "image": "images/bc22d5a245df0f65c673be9d0c974b903fc080cb02d74f7b166f89a95fa6fbba.jpg",
  "audio": "audio/en/bc22d5a245df0f65c673be9d0c974b903fc080cb02d74f7b166f89a95fa6fbba.wav",
  "country": "Egypt",
  "category": "Religion & Spirituality",
  "subcategory": "Holy Texts & Scriptures",
  "answer": "The colorful and intricately designed clothing suggests a connection to traditional cultural practices, possibly indicating their involvement in a performance, storytelling, or a ceremonial activity in the public square."
}
```

---

## Running the Prototype via CLI

The package provides a unified CLI via `python -m mmcultureqa`.

### 1. Inspect Dataset Splits
Inspect records, distribution of cultural categories, and countries:
```bash
python -m mmcultureqa inspect --split train --task qa --lang en
python -m mmcultureqa inspect --split dev --task sqa --lang msa
```

### 2. Run Single-Instance Inference

#### Task 2: Textual Visual QA
```bash
python -m mmcultureqa predict \
  --image data/images/02f8670e06ce58fe8604a7b69b4bd3aec605eaa21e930ac3968d199ddca8b7c4.jpg \
  --question "What can you infer about the cultural significance of the clothing worn by the seated individuals?" \
  --lang en
```

#### Task 1: Spoken Visual QA
Provide the image and audio file; the pipeline will automatically transcribe the audio and reason about the image:
```bash
python -m mmcultureqa predict \
  --image data/images/02f8670e06ce58fe8604a7b69b4bd3aec605eaa21e930ac3968d199ddca8b7c4.jpg \
  --audio data/audio/en/02f8670e06ce58fe8604a7b69b4bd3aec605eaa21e930ac3968d199ddca8b7c4.wav \
  --lang en
```

### 3. Run Batch Predictions
Generate predictions over a whole split using the local offline solver:
```bash
python -m mmcultureqa predict --split dev --task qa --lang en --output dev_preds_en.json
```

Or using the Cultural VLM solver (requires `OPENAI_API_KEY`):
```bash
python -m mmcultureqa predict --split dev --task qa --lang en --solver vlm --model gpt-4o-mini --output dev_vlm_preds_en.json
```

### 4. Benchmark & Evaluate
Evaluate generated predictions against the reference ground truth:
```bash
python -m mmcultureqa evaluate \
  --preds dev_preds_en.json \
  --refs data/qa/dev_en.jsonl \
  --lang en \
  --output evaluation_report_en.json
```

Example output:
```
### MMCultureQA Evaluation Summary (EN)

| Metric                  | Score   | Description                            |
|-------------------------|---------|----------------------------------------|
| BERTScore F1 (Official) | 1.0000  | Semantic similarity (Official Ranking) |
| BERTScore Precision     | 1.0000  | Semantic precision                     |
| BERTScore Recall        | 1.0000  | Semantic recall                        |
| BLEU-4                  | 100.00  | Auxiliary 4-gram lexical overlap       |
| BLEU-1                  | 100.00  | Auxiliary unigram overlap              |
| ROUGE-L                 | 100.00  | Auxiliary longest common subsequence   |
| ROUGE-1                 | 100.00  | Auxiliary unigram recall               |
| Exact Match (%)         | 100.00% | Exact text equality                    |
| Token F1 (%)            | 100.00% | Token overlap F1                       |

### Category Breakdown

| Cultural Category                      | Count | BERTScore F1 | BLEU-4 | ROUGE-L |
|----------------------------------------|-------|--------------|--------|---------|
| Vehicles & Transportation              |     1 |       1.0000 | 100.00 |  100.00 |
| History, Geography & National Identity |     1 |       1.0000 | 100.00 |  100.00 |
| Geography, Buildings & Landmarks       |     1 |       1.0000 | 100.00 |  100.00 |
```

### 5. Package for CodaBench
Create a validated `submission.zip` archive ready to upload to CodaBench:
```bash
python -m mmcultureqa submit --preds dev_preds_en.json --output submission_qa_en.zip
```

---

## Interactive Streamlit Web Dashboard

Launch the browser-based dashboard for interactive exploration:
```bash
streamlit run app.py
```
*(Or via CLI: `python -m mmcultureqa demo`)*

### Dashboard Features:
1. **🎯 Interactive Playground**:
   - Choose between **Task 1 (Spoken QA)** and **Task 2 (Text QA)**.
   - Test on dataset samples or upload custom images and audio WAV recordings.
   - Inspect live ASR audio transcription.
   - Inspect retrieved cultural knowledge entities in real time (Cultural RAG).
   - View latency and immediate metric comparisons against gold references.
2. **📚 Dataset Explorer**:
   - Filter and search records by country, category, subcategory, and language.
   - Listen to spoken questions and browse images.
3. **📊 Benchmark & Evaluation**:
   - Run live evaluations on dev splits and view dynamic breakdown tables by cultural topic and country.
4. **📦 CodaBench Submission Builder**:
   - Single-click packaging and download of competition-ready submission archives.

---

## Evaluation Suite & Metrics

In open-ended cultural visual question answering, rigid token matching metrics (like exact match) severely underestimate system quality because multiple distinct phrasing choices accurately describe the same cultural phenomenon.

Following the SemEval 2027 official specification:
- **Official Ranking Metric**: **BERTScore F1** computes dense contextual embedding cosine similarities between system predictions and references.
- **Auxiliary Metrics**:
  - **SacreBLEU / BLEU (1 to 4)**: Lexical n-gram precision.
  - **ROUGE (1, 2, L)**: Recall-oriented n-gram and longest common subsequence overlap.
  - **Token F1 & Exact Match**: Word-level exact matching metrics for baseline diagnostic context.

---

## Cultural Knowledge Base & RAG

The package includes an extensible cultural grounding module (`mmcultureqa/cultural_knowledge.py`):
1. **Curated Entities**: Covers regional attire (*Galabeya*, *Thobe*, *Bisht*), cultural customs (*Henna*, *Falconry/Bayzara*, *Hakawati* storytelling), architectural monuments (*National Museum of Qatar*, *Desert Rose* formations), and culinary heritage (*Dallah* coffee pots, *Gahwa*, Ramadan *Fanous*).
2. **Cultural Retriever**: Employs multilingual token matching and semantic relevance scoring to retrieve the top cultural concepts relevant to the visual query.
3. **Prompt Injection (Cultural RAG)**: Conditions Vision-Language Models with concise, factual cultural context, enabling accurate answers even when the training set is small or low-resource.

---

## Checkpoints & Rollback History

Every development milestone in this repository is committed with clear, human-readable semantic checkpoints pushed to `main`:

| Checkpoint Commit | Milestone Description |
|---|---|
| `6173e0d` | `chore: initialize repository structure and official sample dataset` |
| `b305b48` | `feat: add dataset loaders and schema validation for OASIS benchmark` |
| `b249f08` | `feat: implement multilingual ASR audio transcription for Task 1` |
| `d7bb4b7` | `feat: introduce cultural knowledge base and grounding retriever` |
| `30fbba1` | `feat: implement multimodal cultural reasoning pipeline and baseline models` |
| `2c0dc42` | `feat: add official BERTScore evaluation suite and CodaBench submission packager` |
| `bd8d8d8` | `feat: provide rich CLI for inspection, inference, and evaluation` |
| `fde85ce` | `feat: add interactive Streamlit web application and visual playground` |

If you ever need to roll back to any stage, run:
```bash
git checkout <commit_hash>
```

---

## Citation

If using this prototype, the OASIS dataset, or participating in the shared task, cite:

```bibtex
@article{alam2025everydaymmqa,
  title = {{OASIS}: A Multilingual and Multimodal Dataset for Culturally Grounded Spoken Visual QA},
  author = {Alam, Firoj and Shahroor, Ali Ezzat and Hasan, Md. Arid and Ali, Zien Sheikh and Bhatti, Hunzalah Hassan and Kmainasi, Mohamed Bayan and Chowdhury, Shammur Absar and Mousi, Basel and Dalvi, Fahim and Durrani, Nadir and Milic-Frayling, Natasa},
  journal = {arXiv preprint arXiv:2510.06371},
  year = {2025},
}
```

---
*Built with ❤️ for the SemEval 2027 Shared Task on Multilingual Multimodal Cultural Question Answering.*
