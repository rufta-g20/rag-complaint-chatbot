# CrediTrust Intelligent Complaint Analysis 🚀

An AI-powered RAG (Retrieval-Augmented Generation) tool built to help CrediTrust Financial transform customer feedback into actionable insights. This tool empowers product managers and compliance teams to query thousands of customer narratives in plain English.

## 📂 Project Structure
As per the CrediTrust technical requirements:
```text
rag-complaint-chatbot/
├── .github/workflows/      # CI/CD pipelines (unittests) 
├── data/
│   ├── raw/                # Original CFPB complaints.csv 
│   └── processed/          # Filtered and cleaned narratives 
├── vector_store/           # Persisted FAISS/ChromaDB indices 
├── notebooks/              # Task 1 EDA and Task 2 experimentation 
├── src/                    # Modular Python logic for RAG pipeline 
├── tests/                  # Unit tests for core functions 
├── app.py                  # Gradio/Streamlit user interface 
├── requirements.txt        # Project dependencies 
└── README.md               # Documentation

```

---

## 📊 Task 1: EDA Summary 

### Data Overview

The initial CFPB dataset contained **9,609,797 total records**. However, the data was highly skewed toward "Credit reporting" services. For our mission at CrediTrust, we filtered the data down to five core product categories: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.

### Key Findings

* **Narrative Availability:** Only **2,980,756 complaints (31%)** contained actual consumer narratives. The remaining 6.6 million records were dropped as they lacked the unstructured text required for RAG.
* **Product Distribution:** After filtering for CrediTrust's core offerings and removing empty narratives, we retained **471,668 high-quality records**. The largest volume of complaints within our scope involves "Checking or savings accounts" and "Credit cards."
* **Narrative Length:** The majority of customer stories are concise, but the distribution shows a long tail, with some narratives exceeding 1,000 words. These long narratives justify our upcoming strategy to use text chunking (500 characters) to ensure the LLM receives relevant context without exceeding token limits.

### Data Preprocessing

To improve embedding quality, all narratives underwent:

1. **Lowercasing:** Standardizing text for consistent vectorization.
2. **Boilerplate Removal:** Stripping common introductory phrases like "I am writing to file a complaint".
3. **Cleaning:** Removing special characters and extra whitespace.

The cleaned dataset is saved at `data/processed/filtered_complaints.csv`.

---

## 🔍 Task 2: Vector Search Infrastructure

### Sampling Strategy
To ensure the RAG system performs efficiently on local hardware while maintaining high-quality responses, I implemented a **stratified sampling strategy**. I selected **14,825 complaints** proportionally across the five core product categories. This prevents the model from being biased toward high-volume categories like Credit Cards and ensures the chatbot can answer questions about Money Transfers or Personal Loans with equal accuracy.

### Text Chunking & Embedding
Long customer narratives were broken down into smaller, manageable pieces to optimize semantic search:
* **Strategy:** Used `RecursiveCharacterTextSplitter`.
* **Chunk Size:** 500 characters.
* **Chunk Overlap:** 50 characters (ensures no context is lost between splits).
* **Total Chunks Created:** 40,991.

### Vector Store & Metadata
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`. I chose this model because it is highly efficient for CPU-based processing (384 dimensions) while providing state-of-the-art semantic mapping for English text.
* **Vector Database:** **FAISS (Facebook AI Similarity Search)**. The index is persisted in `vector_store/faiss_index`.
* **Traceability:** Each vector is stored with metadata including the **Original Complaint ID** and **Product Category**, allowing the RAG pipeline to cite its sources accurately during the retrieval phase.