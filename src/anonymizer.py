"""
Author: Eray Kocabozdoğan
Student ID: 280201055
Anonymization module for PII detection and substitution.
"""

from presidio_analyzer import AnalyzerEngine
from faker import Faker
from transformers import pipeline
import random


class Anonymizer:
    """Handles PII detection and anonymization using multiple strategies."""
    
    def __init__(self):
        print("Initializing Anonymizer Engine...")
        self.analyzer = AnalyzerEngine()
        self.faker = Faker()
        
        print("Loading DistilBERT for Context-Aware substitution...")
        self.fill_mask = pipeline("fill-mask", model="distilbert-base-uncased", device=-1)
        
        self.target_entities = ["PERSON", "GPE", "ORG"] 

    def analyze(self, text):
        """Detect PII entities in text."""
        results = self.analyzer.analyze(text=text, language='en')
        filtered_results = [
            res for res in results 
            if res.entity_type in self.target_entities and res.score > 0.4
        ]
        return filtered_results

    def get_faker_replacement(self, entity_type):
        """Generate synthetic data based on entity type using Faker."""
        if entity_type == "PERSON":
            return self.faker.name()
        elif entity_type in ["GPE", "LOCATION"]:
            return self.faker.city()
        elif entity_type == "ORG":
            return self.faker.company()
        else:
            return "[UNKNOWN]"

    def get_bert_replacement(self, text, start, end, original_word):
        """Generate context-aware replacement using BERT masked language model."""
        masked_text = text[:start] + "[MASK]" + text[end:]
        
        try:
            predictions = self.fill_mask(masked_text, top_k=5)
        except Exception:
            return "[MASKED_ENTITY]"

        best_candidate = "[MASKED_ENTITY]"
        for pred in predictions:
            candidate = pred['token_str'].strip()
            candidate_clean = candidate.lower().replace("Ġ", "")
            original_clean = original_word.lower()

            if (candidate_clean not in original_clean) and \
               (original_clean not in candidate_clean) and \
               (len(candidate_clean) > 2):
                best_candidate = candidate
                break
        
        return best_candidate

    def anonymize(self, text, strategy="placeholder"):
        """
        Anonymize text using specified strategy.
        
        Args:
            text: Input text to anonymize
            strategy: 'placeholder', 'semantic', or 'context_aware'
        
        Returns:
            Anonymized text
        """
        entities = self.analyze(text)
        if not entities:
            return text
            
        entities = sorted(entities, key=lambda x: x.start, reverse=True)
        anonymized_text = text
        
        for entity in entities:
            start = entity.start
            end = entity.end
            entity_type = entity.entity_type
            original_word = text[start:end]
            
            if strategy == "placeholder":
                replacement = f"[{entity_type}]"
            elif strategy == "semantic":
                replacement = self.get_faker_replacement(entity_type)
            elif strategy == "context_aware":
                replacement = self.get_bert_replacement(text, start, end, original_word)
            else:
                replacement = f"[{entity_type}]"

            anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
            
        return anonymized_text