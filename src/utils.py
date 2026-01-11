from datasets import load_dataset

def load_squad_sample(n=10):
    """
    SQuAD veri setinden n tane örnek çeker.
    Hızlı test için n sayısını küçük tut.
    """
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="validation", streaming=True)
    
    samples = []
    count = 0
    for row in dataset:
        samples.append({
            "id": row["id"],
            "context": row["context"],
            "question": row["question"],
            "answers": row["answers"]["text"]
        })
        count += 1
        if count >= n:
            break
            
    return samples