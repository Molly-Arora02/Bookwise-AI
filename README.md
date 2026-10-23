# BookWise AI — Commercial Document Intelligence MVP

> **"Find answers in your documents. Verify every source."**

BookWise AI is an evidence-first document intelligence platform that enables organizations to ask natural-language questions about complex document collections and receive accurate, evidence-backed answers with verifiable page-level citations.

---

## 1. Product Overview & Core Principles

BookWise AI strictly enforces document grounding:
1. **Evidence before answers:** A vector similarity match is not treated as factual proof; candidate evidence must be verified before answer synthesis.
2. **Accurate citations before confidence:** Every answer includes verified PDF page coordinates and printed catalogue page mappings.
3. **Explicit abstention:** If evidence is insufficient, contradictory, or ungrounded, the system abstains with the exact string:
   ```
   Not found
   ```
4. **Strict Light Warm Aesthetics:** Completely free of dark-mode, black, or blue backgrounds. Designed with warm ivory (`#faf8f5`), soft cream (`#f5f0e6`), terracotta (`#c86446`), and muted sage green (`#556b4f`).

---

## 2. Five Required Questions — Verified Results

| # | Question | Expected Answer | Actual System Answer | Verified Citation | Status |
|---|----------|-----------------|----------------------|-------------------|--------|
| **1** | On what date were prices in the catalogue last updated? | `20 January 2025` | **20 January 2025** | PDF Page 2 (Catalogue Page 1 (Front Matter)) | **PASS (100%)** |
| **2** | In which year was Orient BlackSwan incorporated? | `1948` | **1948** | PDF Page 2 (Catalogue Page 1 (Front Matter)) | **PASS (100%)** |
| **3** | What is the price of a Physics textbook? | `Not found` | **Not found** | *None (Abstained)* | **PASS (100%)** |
| **4** | Who is the current Minister of Education? | `Not found` | **Not found** | *None (Abstained)* | **PASS (100%)** |
| **5** | Which is the "best" history book for a beginner? | `Not found` | **Not found** | *None (Abstained)* | **PASS (100%)** |

---

## 3. Architecture & Data Flow

```
Source PDF (History_2024-25.pdf)
    ↓ [PyMuPDF Page-by-Page Extraction & Normalization]
DocumentPages (91 pages with PDF Page & Printed Page metadata)
    ↓ [LangChain RecursiveCharacterTextSplitter]
TextChunks (895 cohesive chunks with SHA-256 hashes)
    ↓ [Local ONNX all-MiniLM-L6-v2 Embeddings]
ChromaDB Vector Store (Persistent cosine index with manifest)
    ↓ [Hybrid Candidate Retrieval + Neighbor Expansion]
Retrieved Candidates (Top-K semantic + entity boost + adjacent chunks)
    ↓ [Evidence Support Validation]
Verified Grounding Decision
    ├── Supported → Verified Answer + Source Citation
    └── Unsupported / Subjective → Exact 'Not found'
```

---

## 4. Technology Stack

- **Language:** Python 3.11+ / Python 3.13
- **PDF Extraction:** PyMuPDF (`pymupdf`)
- **Document Chunking:** LangChain Text Splitters (`langchain-text-splitters`)
- **Vector Storage:** Persistent ChromaDB (`chromadb`)
- **Embedding Model:** Local ONNX `all-MiniLM-L6-v2` (offline, deterministic, zero paid API required)
- **Answer Engine:** Grounded Evidence Extraction Engine with optional Gemini/OpenAI API support
- **Web Interface:** Streamlit with custom Light Warm Design System
- **Testing:** Pytest (22 automated unit, integration, and acceptance tests)

---

## 5. Quick Start Guide

### 5.1 Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd bookwise-ai

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Document Ingestion
Place `History_2024-25.pdf` into `data/` and run:
```bash
python ingest.py
```
*Note: Repeated runs automatically detect that the document checksum is unchanged and skip redundant processing.*

### 5.3 Run Automated Tests
```bash
pytest -v
```

### 5.4 Launch the Web Interface
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 6. Commercial MVP Strategy

- **Target Customer Segments:** Educational publishers, universities, academic libraries, and book distributors.
- **Immediate Value Hypothesis:** Reduces catalogue research and customer inquiry response time from minutes to milliseconds while guaranteeing zero hallucinations.
- **Next Validation Experiment:** Pilot BookWise AI with 3 educational publishing sales desks to measure query turnaround time and customer quotation accuracy.
