# Privacy-Preserving RAG Systems: Evaluating Anonymization Strategies

Author: Eray Kocabozdoğan  
Student ID: 280201055  
Course: CENG543 - Information Retrieval Systems

## Overview

This project investigates how different text anonymization strategies affect the performance of Retrieval-Augmented Generation (RAG) systems. While protecting personally identifiable information (PII) is crucial, anonymization can disrupt semantic structure and harm downstream model performance. We evaluate three anonymization approaches and their impact on retrieval accuracy and answer generation quality.

The research question is: How do different anonymization strategies affect the trade-off between privacy protection and utility preservation in RAG pipelines?

## Anonymization Strategies

We compare three approaches for anonymizing sensitive entities (Person, Organization, Location):

1. **Baseline**: No anonymization (reference).
2. **Placeholder**: Replace entities with generic tags (e.g., `[PERSON]`, `[ORG]`).
3. **Faker**: Replace with synthetic, realistic data (e.g., "Alice" becomes "Jane Smith").
4. **Context-Aware**: Use BERT masked language modeling to generate contextually appropriate replacements.

## Architecture

The system consists of three main components:

1. **Privacy Layer**: PII detection using Presidio Analyzer and spaCy NER.
2. **Anonymization Engine**: Implements the three substitution strategies.
3. **Retrieval Systems**: Dense retrieval using Sentence-Transformers with FAISS or Numpy, and Sparse retrieval using BM25.
4. **Generator**: FLAN-T5-base model for answer generation.

## Dataset

We use SQuAD v1.1 (Stanford Question Answering Dataset) validation split, filtered for PII-rich questions containing keywords like who, where, which organization. The sample size is 500 context-question-answer triplets.

## Installation

Prerequisites: Python 3.8 or higher, 8GB RAM recommended.

Clone the repository and install dependencies:

```bash
git clone https://github.com/eraykocabozdogan/CENG543_Project.git
cd CENG543_Project
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Usage

### 1. Run Main Experiments (Dense + Sparse Retrieval)

```bash
python main.py
```

This runs all four anonymization strategies with both Dense (Numpy) and Sparse (BM25) retrieval methods. Results are saved to `data/results_*.csv`.

### 2. Run FAISS Experiments

```bash
python run_faiss.py
```

This evaluates all strategies using FAISS approximate nearest neighbor search. Results are saved to `faiss_data/results_*_faiss.csv`.

### 3. Filter PII-Rich Rows (Optional but Recommended)

To evaluate the impact of anonymization more accurately, you can filter the dataset to include only rows where PII was actually detected and masked (excluding rows that remained identical to Baseline).

```bash
python filter_pii_rows.py
```

This script:
*   Compares Baseline and Placeholder datasets.
*   Identifies rows where content has changed (PII detected).
*   Generates new `*_filtered.csv` files in `data/` and `faiss_data/` containing 250 PII-rich samples for *all* strategies (Baseline, Placeholder, Faker, Context-Aware).

### 4. Analyze Results

```bash
python analyze_final.py
```

This generates comprehensive evaluation metrics and comparison tables.
*   Automatically detects standard and "Filtered" datasets.
*   Outputs formatted tables to terminal.
*   Generates LaTeX code for papers.
*   Saves results to `final_analysis_results.csv`.

---

## Evaluation Metrics

*   **Retrieval Recall**: Does the retriever find documents containing the correct answer?
*   **Exact Match (EM)**: Does the generated answer exactly match the ground truth?
*   **F1 Score**: Token-level overlap between prediction and ground truth.
*   **Faithfulness**: Is the generated answer grounded in the retrieved context?
*   **Gap (Hallucination)**: Difference between Faithfulness and EM, measures grounded hallucinations.

## Key Findings

Preliminary results show that:
*   **Dataset Impact**: About 43% of the dataset did not contain PII. Filtering for PII-rich rows reveals a much sharper performance drop for anonymization methods.
*   **Performance Drop**: When tested on PII-rich data, Placeholder anonymization causes severe retrieval failures (EM drops from ~50% to **15.2%**).
*   **Comparison**: **Faker** performs even worse than Placeholder in these strict conditions (**14.0% EM**). Context-Aware strategies offer slight improvements (16.0% EM) but still suffer significantly compared to Baseline.
*   **Robustness**: Dense retrieval is generally more robust to anonymization than sparse (BM25) methods.

## Expected Runtime

- `main.py`: ~30-60 minutes
- `run_faiss.py`: ~20-40 minutes
- `analyze_final.py`: < 1 minute

## Plotting
- To reproduce the figures in the paper, run python generate_plots.py.