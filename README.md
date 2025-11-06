# CENG543_Project
CENG543 - Term Project

Title
Evaluating the Impact of Anonymization Strategies on Contextual Integrity and Faithfulness in Retrieval-Augmented Generation

Motivation
While Large Language Models (LLMs) are increasingly used in sensitive domains, they require robust data privacy. Anonymization
is crucial, but its impact on the performance of downstream tasks like Retrieval-Augmented Generation (RAG) is under-explored. 
This project investigates how different anonymization techniques—specifically (A) placeholder substitution (e.g., [PHONE]) 
versus (B) semantic substitution (e.g., format-matching random data)—affect the contextual integrity and faithfulness of RAG 
pipelines. We hypothesize that semantic substitution, while preserving format, may mislead retrievers and degrade factuality 
more than simple placeholders.

Evaluation Plan
We will conduct a comparative evaluation. Retrieval performance will be measured using standard IR metrics (MAP, Recall@K). 
Generation quality will be assessed using metrics beyond simple accuracy, focusing on Faithfulness and Consistency to quantify
the contextual distortion introduced by each anonymization method.
