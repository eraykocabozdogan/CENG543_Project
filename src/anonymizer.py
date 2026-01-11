import spacy
from faker import Faker
from transformers import pipeline
import re

class Anonymizer:
    def __init__(self):
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
        self.faker = Faker()
        
        # YENİ EKLENTİ: Context-Aware (BERT) için model yükleme
        # Sadece bu satır bile projeyi 'advanced' yapar.
        print("Loading Mask-Filling Model (DistilBERT)...")
        self.mask_filler = pipeline("fill-mask", model="distilbert-base-uncased")

    def anonymize(self, text, strategy="placeholder"):
        """
        strategy: 'placeholder', 'semantic' (Faker), veya 'context_aware' (BERT)
        """
        doc = self.nlp(text)
        anonymized_text = text
        
        # Tersten işlem yapıyoruz ki indexler kaymasın
        entities = reversed(list(doc.ents))
        
        for ent in entities:
            label = ent.label_
            
            # Sadece hedef kategoriler
            if label not in ["PERSON", "ORG", "GPE", "DATE"]:
                continue
            
            replacement = ent.text # Varsayılan: Değiştirme

            if strategy == "placeholder":
                replacement = f"[{label}]"
            
            elif strategy == "semantic":
                # Eski yöntem: Rastgele (Faker)
                if label == "PERSON": replacement = self.faker.first_name()
                elif label == "ORG": replacement = self.faker.company()
                elif label == "GPE": replacement = self.faker.city()
                elif label == "DATE": replacement = self.faker.year()
                else: replacement = f"[{label}]"
                
            elif strategy == "context_aware":
                # YENİ YÖNTEM: BERT ile bağlama uygun tahmin
                # Tarihleri BERT ile tahmin etmek zordur, onları Faker'a bırakalım
                if label == "DATE":
                    replacement = self.faker.year()
                else:
                    try:
                        # 1. Entity yerine [MASK] koy
                        masked_sentence = text[:ent.start_char] + "[MASK]" + text[ent.end_char:]
                        
                        # 2. Modeli çalıştır
                        predictions = self.mask_filler(masked_sentence)
                        
                        # 3. Orijinal kelimeyle aynısını seçmemeye çalış
                        best_candidate = predictions[0]['token_str']
                        for pred in predictions:
                            candidate = pred['token_str']
                            # Temizlik: Noktalama işaretlerini ve orijinal kelimeyi ele
                            if candidate.lower() not in ent.text.lower() and len(candidate) > 2:
                                best_candidate = candidate
                                break
                        
                        replacement = best_candidate
                    except Exception as e:
                        # Hata olursa placeholder kullan (Fallback)
                        replacement = f"[{label}]"

            # Metni güncelle
            anonymized_text = anonymized_text[:ent.start_char] + replacement + anonymized_text[ent.end_char:]
            
        return anonymized_text