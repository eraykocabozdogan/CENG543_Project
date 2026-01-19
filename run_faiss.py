"""
Author: Eray Kocabozdoğan
Student ID: 280201055
FAISS-specific experiment runner.
"""

import pandas as pd
import os
import time
from src.utils import load_squad_sample
from src.anonymizer import Anonymizer
from src.rag_pipeline import RAGSystem


def save_faiss_results(results, filename):
    """Save FAISS experiment results to CSV."""
    folder_path = os.path.join(os.getcwd(), "faiss_data")
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, filename)
    
    if not results:
        print(f"   [WARNING] No results to save: {filename}")
        return

    df = pd.DataFrame(results)
    df.to_csv(full_path, index=False)
    print(f"   [SAVED] FAISS results saved to: {full_path}")

def run_experiment_batch(documents, questions, answers, anon_strategy, retrieval_method):
    """Run a batch of FAISS experiments."""
    print(f"\n>>> Running FAISS Experiment: Anonymization='{anon_strategy}'")
    
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
            "generated_answer": model_pred
        })
        
        if (i+1) % 50 == 0:
            print(f"   Processed {i+1}/{len(questions)} queries...")

    duration = time.time() - start_time
    print(f"   Finished in {duration:.2f} seconds.")
    return results

def main():
    """Main FAISS experiment pipeline."""
    NUM_SAMPLES = 500
    
    print("--- Loading Data ---")
    raw_data = load_squad_sample(n=NUM_SAMPLES)
    
    original_docs = [d['context'] for d in raw_data]
    questions = [d['question'] for d in raw_data]
    ground_truths = [d['answers'] for d in raw_data]
    
    anonymizer = Anonymizer()
    
    # Baseline FAISS
    print("\n=== FAISS 1: BASELINE ===")
    res_base = run_experiment_batch(original_docs, questions, ground_truths, "Baseline", "dense_faiss")
    save_faiss_results(res_base, "results_baseline_faiss.csv")

    # Placeholder FAISS
    print("\n=== FAISS 2: PLACEHOLDER ===")
    print("Generating Placeholder data...")
    docs_place = [anonymizer.anonymize(d, strategy="placeholder") for d in original_docs]
    res_place = run_experiment_batch(docs_place, questions, ground_truths, "Placeholder", "dense_faiss")
    save_faiss_results(res_place, "results_placeholder_faiss.csv")

    # Faker FAISS
    print("\n=== FAISS 3: FAKER ===")
    print("Generating Faker data...")
    docs_faker = [anonymizer.anonymize(d, strategy="semantic") for d in original_docs]
    res_faker = run_experiment_batch(docs_faker, questions, ground_truths, "Faker", "dense_faiss")
    save_faiss_results(res_faker, "results_faker_faiss.csv")

    # Context Aware FAISS
    print("\n=== FAISS 4: CONTEXT AWARE ===")
    print("Generating Context Aware data (This may take time)...")
    docs_context = [anonymizer.anonymize(d, strategy="context_aware") for d in original_docs]
    res_context = run_experiment_batch(docs_context, questions, ground_truths, "ContextAware", "dense_faiss")
    save_faiss_results(res_context, "results_context_aware_faiss.csv")

    print("\nAll FAISS experiments completed! Check the 'faiss_data/' folder.")


if __name__ == "__main__":
    main()