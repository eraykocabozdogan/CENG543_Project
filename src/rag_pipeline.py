from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class SimpleRAG:
    def __init__(self):
        print("Loading Retriever Model (MiniLM)...")
        self.retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Loading Generator Model (Flan-T5-Base)...")
        # Flan-T5, dizüstü bilgisayarlarda (CPU) bile hızlı çalışır ve instruction takip eder.
        model_name = "google/flan-t5-base" 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.generator_pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
        
        self.index = None
        self.documents = []

    def ingest_documents(self, documents):
        """
        Dökümanları vektörleştirip FAISS indexine atar.
        """
        self.documents = documents
        print(f"Embedding {len(documents)} documents...")
        embeddings = self.retriever_model.encode(documents)
        
        # FAISS Index oluştur
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))
        print("Indexing complete.")

    def retrieve(self, query, k=1):
        """
        Sorguya en yakın k dökümanı getirir.
        """
        query_vec = self.retriever_model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def generate_answer(self, query, context):
        """
        LLM kullanarak cevap üretir.
        """
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        result = self.generator_pipeline(prompt, max_length=100, do_sample=False)
        return result[0]['generated_text']