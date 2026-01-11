import pandas as pd
import os
from src.utils import load_squad_sample
from src.anonymizer import Anonymizer
from src.rag_pipeline import SimpleRAG

def run_experiment_batch(documents, questions, answers, pipeline_name):
    print(f"\n--- Running Pipeline: {pipeline_name} ---")
    rag = SimpleRAG()
    rag.ingest_documents(documents)
    results = []
    
    for i, question in enumerate(questions):
        retrieved_docs = rag.retrieve(question, k=1)
        context = retrieved_docs[0] if retrieved_docs else ""
        model_prediction = rag.generate_answer(question, context)
        
        results.append({
            "pipeline": pipeline_name,
            "question": question,
            "ground_truth": answers[i],
            "retrieved_context": context,
            "model_answer": model_prediction
        })
        # İlerleme durumunu görmek için log
        print(f"[{pipeline_name}] Q{i+1}/{len(questions)} Processed.")
    return results

def main():
    # --- AYARLAR ---
    # V1 teslimi için 20-30 idealdir. 100 yaparsan işlem çok uzun sürebilir.
    NUM_SAMPLES = 20 
    
    # Çıktı klasörünü belirle ve yoksa oluştur
    OUTPUT_DIR = "data"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"--- STEP 1: Loading {NUM_SAMPLES} samples from SQuAD ---")
    data = load_squad_sample(n=NUM_SAMPLES)
    
    questions = [d['question'] for d in data]
    ground_truths = [d['answers'] for d in data]
    original_docs = [d['context'] for d in data]
    
    # --- STEP 2: Anonymization ---
    print("\n--- STEP 2: Preparing Anonymized Datasets ---")
    anonymizer = Anonymizer()
    
    print("Applying Placeholder Strategy...")
    placeholder_docs = [anonymizer.anonymize(doc, strategy="placeholder") for doc in original_docs]
    
    print("Applying Semantic Strategy (Faker)...")
    semantic_docs = [anonymizer.anonymize(doc, strategy="semantic") for doc in original_docs]
    
    print("Applying Context-Aware Strategy (BERT) - This will take time...")
    context_aware_docs = [anonymizer.anonymize(doc, strategy="context_aware") for doc in original_docs]
    
    # --- STEP 3: Experiments ---
    results_baseline = run_experiment_batch(original_docs, questions, ground_truths, "Baseline")
    results_placeholder = run_experiment_batch(placeholder_docs, questions, ground_truths, "Placeholder")
    results_semantic = run_experiment_batch(semantic_docs, questions, ground_truths, "Semantic (Faker)")
    results_context = run_experiment_batch(context_aware_docs, questions, ground_truths, "Context-Aware (BERT)")
    
    # --- STEP 4: Saving Results ---
    print("\n--- STEP 4: Saving Results ---")
    final_data = []
    for i in range(NUM_SAMPLES):
        row = {
            "Question": questions[i],
            "Ground_Truth": ground_truths[i],
            
            "Baseline_Answer": results_baseline[i]['model_answer'],
            "Placeholder_Answer": results_placeholder[i]['model_answer'],
            "Semantic_Answer": results_semantic[i]['model_answer'],
            "ContextAware_Answer": results_context[i]['model_answer'],

            "Baseline_Context": results_baseline[i]['retrieved_context'],
            "Placeholder_Context": results_placeholder[i]['retrieved_context'],
            "Semantic_Context": results_semantic[i]['retrieved_context'],
            "ContextAware_Context": results_context[i]['retrieved_context']
        }
        final_data.append(row)
        
    df = pd.DataFrame(final_data)
    
    # Dosya yollarını data/ klasörüne yönlendir
    excel_path = os.path.join(OUTPUT_DIR, "experiment_results_v1.xlsx")
    csv_path = os.path.join(OUTPUT_DIR, "experiment_results_v1.csv")
    
    df.to_excel(excel_path, index=False)
    df.to_csv(csv_path, index=False)
    
    print(f"Successfully saved results to:\n -> {excel_path}\n -> {csv_path}")

if __name__ == "__main__":
    main()