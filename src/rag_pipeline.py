"""
Author: Eray Kocabozdoğan
Student ID: 280201055
RAG (Retrieval-Augmented Generation) System implementation.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class RAGSystem:
    """RAG system supporting multiple retrieval methods."""
    
    def __init__(self, retrieval_method='dense_faiss', model_name="google/flan-t5-base"):
        """
        Initialize RAG system.
        
        Args:
            retrieval_method: 'dense_faiss', 'dense_numpy', or 'sparse_bm25'
            model_name: Hugging Face model name for text generation
        """
        self.retrieval_method = retrieval_method
        self.documents = []
        
        print(f"Loading Generator Model ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.generator = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

        if 'dense' in retrieval_method:
            print("Loading Embedder (MiniLM) for Dense Retrieval...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = None
            self.doc_embeddings = None
        elif retrieval_method == 'sparse_bm25':
            print("Initializing BM25 for Sparse Retrieval...")
            self.bm25 = None
            
    def ingest_documents(self, documents):
        """Index documents using the selected retrieval method."""
        self.documents = documents
        
        if self.retrieval_method == 'dense_faiss':
            print(f"Ingesting {len(documents)} documents into FAISS...")
            embeddings = self.embedder.encode(documents)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            
        elif self.retrieval_method == 'dense_numpy':
            print(f"Encoding {len(documents)} documents for Numpy Exact Search...")
            self.doc_embeddings = self.embedder.encode(documents)
            
        elif self.retrieval_method == 'sparse_bm25':
            print(f"Tokenizing {len(documents)} documents for BM25...")
            tokenized_corpus = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query, k=1):
        """Retrieve top-k documents most relevant to the query."""
        if self.retrieval_method == 'dense_faiss':
            query_vec = self.embedder.encode([query])
            distances, indices = self.index.search(np.array(query_vec).astype('float32'), k)
            valid_indices = [idx for idx in indices[0] if idx < len(self.documents) and idx >= 0]
            return [self.documents[idx] for idx in valid_indices]
            
        elif self.retrieval_method == 'dense_numpy':
            query_vec = self.embedder.encode([query])
            scores = cosine_similarity(query_vec, self.doc_embeddings)[0]
            top_k_indices = np.argsort(scores)[::-1][:k]
            return [self.documents[idx] for idx in top_k_indices]
            
        elif self.retrieval_method == 'sparse_bm25':
            tokenized_query = query.lower().split()
            return self.bm25.get_top_n(tokenized_query, self.documents, n=k)

    def generate_answer(self, query, context):
        """Generate answer using LLM based on retrieved context and query."""
        input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        results = self.generator(input_text, max_length=64, do_sample=False)
        return results[0]['generated_text']