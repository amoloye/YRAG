# RAG-Based PDF Retrieval System (spaCy + HNSW)
## Overview

This project implements a lightweight **Retrieval-Augmented Generation (RAG)** pipeline for semantic search over PDF documents. The system extracts text from PDFs, generates vector embeddings using spaCy, indexes them with HNSW, and retrieves relevant passages for language model–based answer generation.

The implementation prioritises reproducibility, CPU compatibility, and minimal dependency overhead.

---------------------------------------------------------------------------------------------------------------------------------

## Motivation

Many vector search pipelines rely on heavy machine learning frameworks such as **PyTorch** (e.g., Sentence-Transformers, FAISS, ChromaDB). These can introduce compatibility and installation issues, particularly on macOS Intel environments.

This project intentionally uses:

- **spaCy (en_core_web_md)** for static word embeddings

- **hnswlib (HNSW)** for approximate nearest neighbour search

This approach avoids GPU requirements and large deep learning dependencies while maintaining functional semantic retrieval.

---------------------------------------------------------------------------------------------------------------------------------

## Methodology

The pipeline consists of the following stages:

- ### Document Ingestion

  - PDF parsing via PyMuPDF

  - Page-level text extraction

- ### Chunking

  - Sentence-aware segmentation

  - Maximum character constraint per chunk

- ### Embedding

  - spaCy en_core_web_md

  - 300-dimensional static word vectors

  - Mean pooling over document tokens

- ### Indexing

  - HNSW (Hierarchical Navigable Small World) graph

  - Cosine similarity space

  - Persistent index storage

- ### Retrieval-Augmented Generation

  - Query embedding

  - Top-k nearest neighbour retrieval

  - Context assembly

- ### Response generation via Groq LLM

---------------------------------------------------------------------------------------------------------------------------------

## System Architecture

PDF → Text Extraction → Chunking → Embedding → HNSW Index
                                                ↓
                                           Retrieval
                                                ↓
                                       LLM Response (RAG)

---------------------------------------------------------------------------------------------------------------------------------                                      

## Technical Design Choices
- spaCy Embeddings

- 300-dimensional static vectors

- No PyTorch dependency

- Fully CPU-compatible

- Stable across macOS Intel environments

---------------------------------------------------------------------------------------------------------------------------------

## Trade-off:

- Lower semantic performance compared to transformer-based embeddings

- Chosen for simplicity and system stability

- HNSW (hnswlib)

- Approximate nearest neighbour search

- Logarithmic retrieval complexity

- Memory-efficient

- CPU-based

- Supports persistence

- HNSW provides scalable semantic search without requiring FAISS or GPU acceleration.

---------------------------------------------------------------------------------------------------------------------------------

## Example Query

### Input:

"What is loadbearing?"


### Process:

- Embed query

- Retrieve top-k similar chunks

- Provide retrieved context to LLM

- Generate a concise answer grounded in document content

---------------------------------------------------------------------------------------------------------------------------------

## Dependencies

- Core libraries:

- Python 3.13

- spaCy

- hnswlib

- PyMuPDF

- NumPy

- LangChain

- Groq API

---------------------------------------------------------------------------------------------------------------------------------

## Limitations

- Static embeddings (no contextual transformer encoding)

- Retrieval quality is dependent on chunk granularity

- No reranking or hybrid search implemented

---------------------------------------------------------------------------------------------------------------------------------

## Future Work

- Replace static embeddings with lightweight transformer embeddings

- Add hybrid keyword + vector search

- Implement cross-encoder reranking

- Benchmark retrieval performance quantitatively

---------------------------------------------------------------------------------------------------------------------------------

## Image Generated


<img width="834" height="716" alt="Screenshot 2026-02-11 at 14 40 07" src="https://github.com/user-attachments/assets/f15e0328-bd22-4163-ae0a-e7541384179b" />

