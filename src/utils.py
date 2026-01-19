"""
Author: Eray Kocabozdoğan
Student ID: 280201055
Utility functions for data loading and result saving.
"""

import json
from datasets import load_dataset


def load_squad_sample(n=500):
    """
    Load n samples from SQuAD v1.1 validation set.
    Filters for questions likely to contain PII (who, where, which company, etc.).
    
    Args:
        n: Number of samples to load
    
    Returns:
        List of dictionaries containing context, question, and answer
    """
    print(f"Loading SQuAD v1.1 validation set (Targeting PII-heavy questions)...")
    data = load_dataset("squad", split="validation")
    
    samples = []
    target_triggers = ["who", "where", "which company", "which organization", "which city", "name of"]
    
    for item in data:
        question_text = item['question'].lower()
        
        if any(trigger in question_text for trigger in target_triggers):
            context = item['context']
            question = item['question']
            answer = item['answers']['text'][0] if item['answers']['text'] else ""
            
            samples.append({
                "context": context,
                "question": question,
                "answers": answer
            })
            
        if len(samples) >= n:
            break
            
    print(f"Loaded {len(samples)} filtered samples.")
    return samples

def save_results(results, filename="data/experiment_results_final.csv"):
    """Save experiment results to CSV file."""
    import pandas as pd
    import os
    
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")