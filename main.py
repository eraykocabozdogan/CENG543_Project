"""
Author: Eray Kocabozdoğan
Student ID: 280201055
CENG543 Term Project - Main Experiment Runner
"""

import pandas as pd
import os
import time
from src.utils import load_squad_sample
from src.anonymizer import Anonymizer
from src.rag_pipeline import RAGSystem


def save_batch_results(results, filename):
    """Save experiment results to CSV file."""
    folder_path = os.path.join(os.getcwd(), "data")
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, filename)
    
    if not results:
        print(f"   [WARNING] No results to save: {filename}")
        return

    df = pd.DataFrame(results)
    df.to_csv(full_path, index=False)
    print(f"   [SAVED] Results saved to: {full_path}")

def run_experiment_batch(documents, questions, answers, anon_strategy, retrieval_method):
    """Run a batch of RAG experiments with specified anonymization and retrieval method."""
    print(f"\n>>> Running Experiment: Anonymization='{anon_strategy}' | Retrieval='{retrieval_method}'")
    
    rag = RAGSystem(retrieval_method=retrieval_method)
    rag.ingest_documents(documents)
    
    results = []
    start_time = time.time()
    
    for i, (q, truth) in enumerate(zip(questions, answers)):
        retrieved_docs = rag.retrieve(q, k=1)
        context = retrieved_docs[0] if retrieved_docs else ""
        model_pred = rag.generate_answer(q, context)
        
        results.append({
            "anonymization_strategy": anon_strategy,
            "retrieval_method": retrieval_method,
            "question": q,
            "ground_truth": truth,
            "retrieved_context_snippet": context[:200], 
            "model_answer": model_pred
        })
        
        if (i+1) % 50 == 0:
            print(f"   Processed {i+1}/{len(questions)} queries...")

    duration = time.time() - start_time
    print(f"   Finished in {duration:.2f} seconds.")
    return results

def main():
    """Main experiment pipeline."""
    NUM_SAMPLES = 500
    
    raw_data = load_squad_sample(n=NUM_SAMPLES)
    original_docs = [d['context'] for d in raw_data]
    questions = [d['question'] for d in raw_data]
    ground_truths = [d['answers'] for d in raw_data]
    
    anonymizer = Anonymizer()
    print("\n--- Preparing Anonymized Datasets ---")

    # Scenario 1: Baseline (No Anonymization)
    print("\n=== SCENARIO 1: BASELINE ===")
    results_baseline = []
    results_baseline.extend(run_experiment_batch(original_docs, questions, ground_truths, "Baseline", "dense_numpy"))
    results_baseline.extend(run_experiment_batch(original_docs, questions, ground_truths, "Baseline", "sparse_bm25"))
    save_batch_results(results_baseline, "results_01_baseline.csv")


    # Scenario 2: Placeholder Anonymization
    print("\n=== SCENARIO 2: PLACEHOLDER ===")
    print("Generating Placeholder dataset...")
    docs_placeholder = [anonymizer.anonymize(d, strategy="placeholder") for d in original_docs]
    
    results_placeholder = []
    results_placeholder.extend(run_experiment_batch(docs_placeholder, questions, ground_truths, "Placeholder", "dense_numpy"))
    results_placeholder.extend(run_experiment_batch(docs_placeholder, questions, ground_truths, "Placeholder", "sparse_bm25"))
    save_batch_results(results_placeholder, "results_02_placeholder.csv")


    # Scenario 3: Faker (Semantic Substitution)
    print("\n=== SCENARIO 3: FAKER (SEMANTIC) ===")
    print("Generating Faker dataset...")
    docs_faker = [anonymizer.anonymize(d, strategy="semantic") for d in original_docs]
    
    results_faker = []
    results_faker.extend(run_experiment_batch(docs_faker, questions, ground_truths, "Faker", "dense_numpy"))
    results_faker.extend(run_experiment_batch(docs_faker, questions, ground_truths, "Faker", "sparse_bm25"))
    save_batch_results(results_faker, "results_03_faker.csv")


    # Scenario 4: Context-Aware (BERT-based)
    print("\n=== SCENARIO 4: CONTEXT AWARE ===")
    print("Generating Context-Aware dataset (This takes time)...")
    docs_context = [anonymizer.anonymize(d, strategy="context_aware") for d in original_docs]
    
    results_context = []
    results_context.extend(run_experiment_batch(docs_context, questions, ground_truths, "ContextAware", "dense_numpy"))
    save_batch_results(results_context, "results_04_context_aware.csv")

    print("\nAll experiments completed! Check the 'data/' folder.")

if __name__ == "__main__":
    main()