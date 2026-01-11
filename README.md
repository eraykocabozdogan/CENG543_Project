# CENG543 Term Project

# Evaluating the Impact of Anonymization Strategies on Contextual Integrity and Faithfulness in Retrieval-Augmented Generation

## Overview
This project investigates the trade-off between data privacy and utility in Retrieval-Augmented Generation (RAG) systems. While Large Language Models require vast amounts of data, using sensitive information raises privacy concerns. We focus on how different anonymization techniques affect the downstream performance of RAG pipelines, specifically analyzing the impact on retrieval accuracy and the faithfulness of generated answers.

## Problem Statement
Standard anonymization techniques often disrupt the semantic structure of text. We compare three primary strategies to understand their effects:

1. **Placeholder Substitution:** Replacing sensitive entities with generic tags (e.g., `[PERSON]`, `[LOCATION]`).
2. **Semantic Substitution (Faker):** Replacing entities with synthetic, format-preserving data (e.g., replacing "Alice" with "Jane").
3. **Context-Aware Substitution (BERT):** Using a Masked Language Model to generate synthetic entities that fit the semantic context of the sentence (Utility-Preserving Anonymization).

## Methodology
The experimental pipeline consists of three main modules:

- **Privacy Preservation Layer:** We employ NLP-based entity detection (spaCy) to identify PII. The identified entities are then processed using the three substitution strategies mentioned above.
- **Retrieval System:** Instead of sparse retrieval methods, we utilize a Dense Passage Retriever (Bi-Encoder) architecture based on Sentence-Transformers (`all-MiniLM-L6-v2`) to index and search the anonymized contexts.
- **Generation:** An open-source instruction-tuned LLM (`google/flan-t5-base`) is used to generate answers based on the retrieved, anonymized contexts.

## Dataset
We utilize the **Stanford Question Answering Dataset (SQuAD v1.1)** for this study. SQuAD provides high-quality context-question-answer triplets, allowing for a controlled evaluation of reading comprehension. We synthetically inject and then anonymize PII (Personally Identifiable Information) within the contexts to simulate sensitive documents. The target PII categories include Person, Location, Organization, and Date.

## Installation & Usage

To reproduce the experiments, please follow the steps below:

### 1. Prerequisites
- Python 3.8 or higher
- pip

### 2. Installation
Clone the repository and install the required dependencies:

``bash
# Install dependencies
pip install -r requirements.txt

# Download necessary spaCy model
"python -m spacy download en_core_web_sm"

### 3. Running the Experiments

Run the main script to execute the full pipeline (Data Loading -> Anonymization -> RAG Retrieval & Generation):

"python main.py"

### 4. Outputs

After the execution is complete, the results will be saved in the data/ directory:

- data/experiment_results_v1.csv: Detailed CSV report containing questions, ground truths, retrieved contexts, and generated answers for all strategies.

- data/experiment_results_v1.xlsx: Excel format of the results.

### Preliminary Results
Initial experiments indicate that while placeholder substitution leads to context loss and retrieval failures, semantic substitution introduces "grounded hallucinations," where the model generates factually incorrect answers with high confidence based on synthetic entities. Context-aware substitution aims to mitigate these issues by preserving semantic coherence.