CENG543 Term Project
Title
Evaluating the Impact of Anonymization Strategies on Contextual Integrity and Faithfulness in Retrieval-Augmented Generation

Overview
This project investigates the trade-off between data privacy and utility in Retrieval-Augmented Generation (RAG) systems. While Large Language Models require vast amounts of data, using sensitive information raises privacy concerns. We focus on how different anonymization techniques affect the downstream performance of RAG pipelines, specifically analyzing the impact on retrieval accuracy and the faithfulness of generated answers.

Problem Statement
Standard anonymization techniques often disrupt the semantic structure of text. We compare two primary strategies to understand their effects: (A) Placeholder Substitution: Replacing sensitive entities with generic tags (e.g., [PERSON], [LOCATION]). (B) Semantic Substitution: Replacing entities with synthetic, format-preserving data (e.g., replacing "Alice" with "Jane", or a real address with a fake but valid address format).

We hypothesize that while semantic substitution preserves linguistic fluency, it may introduce false semantic signals that mislead dense retrievers, whereas placeholder substitution might degrade the context but preserve structural integrity.

Dataset
We utilize the Stanford Question Answering Dataset (SQuAD v1.1) for this study. SQuAD provides high-quality context-question-answer triplets, allowing for a controlled evaluation of reading comprehension. We synthetically inject and then anonymize PII (Personally Identifiable Information) within the contexts to simulate sensitive documents. The target PII categories include Person, Location, Organization, Date, and Phone/Email.

Methodology
The experimental pipeline consists of three main modules:

Privacy Preservation Layer: We employ NLP-based entity detection to identify PII. The identified entities are then processed using the two substitution strategies mentioned above.

Retrieval System: Instead of sparse retrieval methods, we utilize a Dense Passage Retriever (Bi-Encoder) architecture based on Sentence-Transformers (e.g., all-MiniLM) to index and search the anonymized contexts.

Generation: An open-source instruction-tuned LLM (e.g., Mistral or Llama families) is used to generate answers based on the retrieved, anonymized contexts.

Evaluation Plan
We conduct a comparative analysis between the baseline (non-anonymized) system and the two anonymization strategies.

Retrieval Performance: Measured using standard Information Retrieval metrics such as Recall@K and Mean Reciprocal Rank (MRR) to assess if the correct documents are retrieved despite the data perturbation.

Generation Quality: Evaluated using SQuAD standard metrics (Exact Match, F1-Score) and specific RAG metrics like Faithfulness to quantify how much the anonymization process distorts the factual grounding of the answers.
