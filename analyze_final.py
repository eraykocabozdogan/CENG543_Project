"""
Author: Eray Kocabozdoğan
Student ID: 280201055
Final analysis script for evaluating experiment results.
"""

import pandas as pd
import glob
import os
import re
import string
from collections import Counter


def normalize_answer(s):
    """Normalize text: lowercase, remove punctuation and extra whitespace."""
    if not isinstance(s, str): 
        return ""
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """Calculate token-level F1 score (SQuAD standard)."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def calculate_all_metrics(df):
    """
    Calculate five core metrics for all rows in DataFrame:
    1. Retrieval Recall
    2. Exact Match (EM)
    3. F1 Score
    4. Faithfulness
    5. Gap (Grounded Hallucination Rate)
    """
    total = len(df)
    if total == 0:
        return [0]*5
    
    em_scores = []
    f1_scores = []
    faith_count = 0
    recall_count = 0
    
    for _, row in df.iterrows():
        pred = str(row.get('model_answer') or row.get('generated_answer') or "")
        
        truth_raw = row.get('ground_truth')
        truth = str(truth_raw).replace("['", "").replace("']", "").replace('["', '').replace('"]', '').split("', '")[0]
        context = str(row.get('retrieved_context_snippet') or "")
        
        norm_pred = normalize_answer(pred)
        norm_truth = normalize_answer(truth)
        norm_context = normalize_answer(context)
        
        # Exact Match
        em_scores.append(1 if norm_pred == norm_truth else 0)
        
        # F1 Score
        f1_scores.append(f1_score(pred, truth))
        
        # Faithfulness: Is prediction grounded in context?
        if norm_pred and norm_pred in norm_context:
            faith_count += 1
            
        # Retrieval Recall: Is ground truth in retrieved context?
        if norm_truth and norm_truth in norm_context:
            recall_count += 1
    # Calculate averages (as percentages)
    avg_recall = (recall_count / total) * 100
    avg_em = (sum(em_scores) / total) * 100
    avg_f1 = (sum(f1_scores) / total) * 100
    avg_faith = (faith_count / total) * 100
    
    # Gap: Grounded Hallucination Rate
    # Model is faithful to text but doesn't provide correct answer
    gap = avg_faith - avg_em
    
    return avg_recall, avg_em, avg_f1, avg_faith, gap


def main():
    """Analyze all experiment results and generate summary tables."""
    files = glob.glob("data/*.csv") + glob.glob("faiss_data/*.csv")
    
    results = []
    print(f"--- Analyzing {len(files)} files ---\n")
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            
            # Determine anonymization strategy from filename
            # Determine anonymization strategy from filename
            fname = os.path.basename(f).lower()
            suffix = " (Filtered)" if "filtered" in fname else ""
            
            if "baseline" in fname:
                anon = f"Baseline{suffix}"
            elif "placeholder" in fname:
                anon = f"Placeholder{suffix}"
            elif "faker" in fname:
                anon = f"Faker{suffix}"
            elif "context_aware" in fname:
                anon = f"Context-Aware{suffix}"
            else:
                anon = f"Unknown{suffix}"
            # Determine architecture (retrieval method)
            
            # FAISS files
            if "faiss" in fname or "faiss" in f:
                recall, em, f1, faith, gap = calculate_all_metrics(df)
                results.append({
                    "Architecture": "Dense (FAISS)",
                    "Anonymization": anon,
                    "Retrieval Recall": recall,
                    "Exact Match (EM)": em,
                    "F1 Score": f1,
                    "Faithfulness": faith,
                    "Gap (Hallucination)": gap
                })
            
            # Numpy and BM25 files
            elif 'retrieval_method' in df.columns:
                groups = df.groupby('retrieval_method')
                for method, group_df in groups:
                    if "numpy" in method:
                        arch = "Dense (Exact)"
                    elif "bm25" in method:
                        arch = "Sparse (BM25)"
                    else:
                        arch = method
                    
                    recall, em, f1, faith, gap = calculate_all_metrics(group_df)
                    results.append({
                        "Architecture": arch,
                        "Anonymization": anon,
                        "Retrieval Recall": recall,
                        "Exact Match (EM)": em,
                        "F1 Score": f1,
                        "Faithfulness": faith,
                        "Gap (Hallucination)": gap
                    })
                    
        except Exception as e:
            print(f"Error ({f}): {e}")

    # Format and output results
    df_res = pd.DataFrame(results)
    
    if df_res.empty:
        print("Error: No results generated! Check folder paths.")
        return

    # Sort by Architecture and Anonymization
    arch_order = ["Sparse (BM25)", "Dense (Exact)", "Dense (FAISS)"]
    anon_order = [
        "Baseline", "Baseline (Filtered)", 
        "Placeholder", "Placeholder (Filtered)", 
        "Faker", "Faker (Filtered)", 
        "Context-Aware", "Context-Aware (Filtered)",
        "Unknown", "Unknown (Filtered)"
    ]
    
    df_res['Architecture'] = pd.Categorical(df_res['Architecture'], categories=arch_order, ordered=True)
    df_res['Anonymization'] = pd.Categorical(df_res['Anonymization'], categories=anon_order, ordered=True)
    df_res = df_res.sort_values(by=["Architecture", "Anonymization"])
    
    # Terminal output
    print("\n" + "="*100)
    print("PROJECT ANALYSIS RESULTS (FINAL)")
    print("="*100)
    print(df_res.to_string(index=False, float_format="%.1f"))
    
    # LaTeX output
    print("\n" + "="*100)
    print("LATEX TABLE CODE (For paper)")
    print("="*100)
    print(df_res.to_latex(index=False, float_format="%.1f"))
    
    # Save to CSV
    df_res.to_csv("final_analysis_results.csv", index=False)
    print("\n[INFO] Results saved to 'final_analysis_results.csv'.")


if __name__ == "__main__":
    main()
