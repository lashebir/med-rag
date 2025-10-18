# **High-Level Overview:**

## This Retrieval-Augmented Generation (RAG) system has been created to perform a multi-source semantic search on topics related to neurological (i.e. sensironeural) hearing loss.

## Built for educational purposes, the application integrates a Fast API backend framework with a PostGreSQL + pgvector database to index and embed research literature from multiple open-source platforms (e.g. PubMed Central, arXiv).

## In the final design, the system will first be prompted by the client to find relevant research material, which will trigger the framework to search the vector database for a match, rank matches across the multiple sources, and return these sources to the client via a large language model (LLM) in a coherent, human friendly summary.

# **High-Level Architecture:**

## * **FastAPI** for serving and managing user queries

## * **PostgreSQL + pgvector** for semantic search and vector storage

## * **External APIs** (PubMed Central, arXiv, etc.) for document ingestion

## * **LLM** integration for context-aware synthesis of retrieved results