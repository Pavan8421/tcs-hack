# AI-Powered Insurance Policy Information Chatbot Documentation

## Executive Summary

This document describes an AI-powered Insurance Policy Information Chatbot designed to assist customers with queries about insurance policies. The system leverages large language models (LLMs), vector search, and natural language processing to provide accurate and contextually relevant responses based on insurance policy documents. The chatbot allows users to either ask questions about insurance policies or upload new policy documents to expand the knowledge base.

## System Overview

The chatbot combines document processing, semantic search, and generative AI technologies to create an intelligent assistant that can understand policy documents and answer user questions about them. The architecture consists of several key components working together to provide a seamless experience.

**Code Repository:** [https://github.com/Pavan8421/tcs-hack](https://github.com/Pavan8421/tcs-hack)

**Flow chart:**
![alt text](https://drive.google.com/uc?export=view&id=1tXIMzZa1FMjJSatdkaZ8PkiknmfVmoID)


## Core Components

### 1. Knowledge Base Construction

The system builds and maintains a knowledge base of insurance policy information through:

- **PDF Text Extraction**: Using PyMuPDF to extract raw text from uploaded policy documents
- **Paragraph Chunking**: Splitting documents into manageable sections based on natural paragraph breaks
- **Semantic Enrichment**: Processing each chunk with the Zephyr-7B language model to extract structured information
- **Vector Embedding**: Converting text into numerical representations using the BGE embedding model
- **FAISS Indexing**: Storing vector embeddings in a FAISS index for efficient similarity search

### 2. Query Processing Pipeline

When a user asks a question, the system:

- **Embeds the Query**: Converts the user's question into a vector representation
- **Performs Hybrid Search**: Combines dense vector search with keyword matching for improved relevance
- **Retrieves Context**: Identifies the most relevant document sections from the knowledge base
- **Generates Response**: Uses Mistral-7B to craft a natural language response based on the retrieved context

### 3. User Interface

The system provides a streamlined interface that:

- Accepts natural language questions from users
- Allows uploading of new policy documents to expand the knowledge base
- Displays clear, concise responses to user queries
- Manages the conversation flow with appropriate status indicators

## Technical Architecture

### Model Selection

- **Mistral-7B-Instruct-v0.2**: Powers the response generation, converting retrieved context into natural language answers
- **HuggingFaceH4/zephyr-7b-beta**: Processes and enriches document chunks, extracting structured information
- **BAAI/bge-base-en-v1.5**: Creates high-quality embeddings for both documents and queries

### Search Methodology

The system employs a hybrid search approach that combines:

- **Dense Retrieval**: Vector similarity using FAISS for semantic understanding
- **Sparse Retrieval**: Keyword matching for factual precision
- **Weighted Combination**: Balancing semantic and lexical relevance for optimal results

### Knowledge Processing Flow

1. Document Upload & Processing
   - PDF text extraction
   - Text chunking
   - Structured information extraction
   - Vector representation
   - Index updating

2. Query Processing
   - Query embedding
   - Hybrid search execution
   - Context compilation
   - Response generation
   - User presentation

## Implementation Details

### Knowledge Enrichment Process

The system enriches raw text by extracting structured information including:
- Policy type (health, auto, life, etc.)
- Coverage details and limitations
- Key content summaries in natural language

This enrichment improves search relevance and enables more precise answers.

### Hybrid Search Algorithm

The hybrid search combines:
- Vector similarity scores from the FAISS index
- Keyword overlap between query and documents
- A weighted blend of both scores to balance semantic understanding with factual precision

### Error Handling and Fallbacks

The system implements multiple fallback mechanisms:
- JSON parsing fallback using regex extraction when LLM parsing fails
- Graceful error handling with appropriate user notifications
- Content validation to ensure quality responses

## Deployment Architecture

The application is built on Streamlit, providing:
- Interactive web interface
- Real-time response generation
- Document upload capabilities
- Comprehensive logging system

## Limitations and Future Enhancements

Current limitations:
- Processing speed for larger documents
- Handling of complex policy language
- Limited visualization of policy details

Potential enhancements:
- **Optical Character Recognition (OCR)**: Implementing advanced OCR technologies to improve text extraction accuracy from scanned documents, poor quality PDFs, and image-based policy documents
- **IOB (Inside-Outside-Beginning) Tagging**: Utilizing named entity recognition techniques with IOB tagging for more precise information extraction from policy documents, allowing better identification of policy terms, exclusions, and conditions
- **Advanced Chunking Methods**: Moving beyond paragraph-based chunking to semantic chunking that preserves context relationships, implementing sliding window approaches with overlap, and hierarchical chunking that maintains document structure
- Multi-modal capabilities to process images and charts within policies
- Personalized responses based on user history
- More sophisticated document structure analysis
- Integration with external policy databases and resources
- Improved handling of numerical data like coverage amounts and premiums
- Automatic categorization of policy documents using classification models

## Conclusion

The AI-Powered Insurance Policy Information Chatbot successfully addresses the need for intuitive access to insurance policy information. By combining advanced language models with efficient vector search techniques, the system enables users to extract relevant information from complex policy documents through natural conversation. The ability to continuously expand its knowledge base through document uploads ensures the chatbot remains current and comprehensive in its coverage of insurance policies.

**Source Code:** The complete implementation is available at [https://github.com/Pavan8421/tcs-hack](https://github.com/Pavan8421/tcs-hack)
