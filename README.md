## **High-Level Overview:**

This Retrieval-Augmented Generation (RAG) system has been created to perform a multi-source semantic search on topics related to neurological (i.e. sensironeural) hearing loss.

Built for educational purposes, the application integrates a Fast API backend framework with a PostGreSQL + pgvector database to index and embed research literature from multiple open-source platforms (e.g. PubMed Central, arXiv).

In the final design, the system will first be prompted by the client to find relevant research material, which will trigger the framework to search the vector database for a match, rank matches across the multiple sources, and return these sources to the client via a large language model (LLM) in a coherent, human friendly summary.

## **High-Level Architecture:**

* **FastAPI** for serving and managing user queries

* **PostgreSQL + pgvector** for semantic search and vector storage

* **External APIs** (PubMed Central, arXiv, Google Scholar) for document ingestion

* **LLM** (Ollama) integration for context-aware synthesis of retrieved results

## **Documentation:**

### Search & Retrieval
* [IVFFlat Tuning Guide](app/search_strategies/IVFFLAT_TUNING.md) - Vector index configuration and performance tuning
* [Search Strategies Overview](app/search_strategies/SEARCH_STRATEGIES.md) - Comparison of 4 retrieval strategies
* [Strategy 4 (Full Hybrid)](app/search_strategies/STRATEGY_4_SUMMARY.md) - IVFFlat + NER + tsvector implementation
* [Embedding Superposition Analysis](app/search_strategies/IVFFlat_analysis/EMBEDDING_SUPERPOSITION_ANALYSIS.md) - Feature entanglement and disentanglement in vector space

### A/B Testing & Evaluation
* [A/B Testing Guide](app/AB_TESTING_GUIDE.md) - Production testing framework for comparing search strategies
* [Benchmarking Script](app/search_strategies/benchmark_search_strategies.py) - Offline performance evaluation

### Data Ingestion & Enrichment
* [Auto DOI Enrichment](app/ingestion/DOI/AUTO_DOI_ENRICHMENT.md) - Automatic DOI extraction for newly ingested documents
* [NER Backfill Script](app/ingestion/NER/backfill_ner.py) - Entity extraction for existing documents