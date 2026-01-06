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

## ⚙️ Getting Started

### 1. Environment Setup

Create a virtual environment and install the required dependencies:

```bash
# Create environment
python -m venv creditrust_env

# Activate (Windows)
.\creditrust_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Configuration & Environment Variables

The system is designed to work out-of-the-box, but you can configure the following:

* **Hugging Face Cache:** If you have limited disk space on `C:`, set the environment variable `HF_HOME` to a different drive.
* **Developer Mode:** On Windows, it is recommended to enable **Developer Mode** to allow the system to create symlinks for the embedding models.

### 3. Execution Steps

Run the following commands in order to build the pipeline:

* **Task 1: Preprocessing & EDA**: Execute the provided notebook to filter the raw data, perform exploratory analysis, and generate the cleaned dataset.
```bash
# Open and run all cells in the preprocessing notebook
jupyter notebook notebooks/preprocessing.ipynb

```

* **Task 2: Indexing** (Generate embeddings and FAISS index)
```bash
# Parameters like --sample_size or --chunk_size can be adjusted
python -m src.vector_indexing --sample_size 15000 --chunk_size 500

```

---

## 📊 Task 1: EDA Summary

### Data Overview

The initial CFPB dataset contained **9,609,797 total records**. However, the data was highly skewed toward "Credit reporting" services. For our mission at CrediTrust, we filtered the data down to five core product categories: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.

### Key Findings

* **Narrative Availability:** Only **31% of complaints** contained consumer narratives. Records without narratives were dropped as they lacked the unstructured text required for RAG.
* **Product Distribution:** After filtering for CrediTrust's core offerings and removing empty narratives, we retained **471,668 high-quality records**. The largest volume of complaints within our scope involves "Checking or savings accounts" and "Credit cards".
* **Narrative Length:** The majority of narratives are concise, but a long tail exists with narratives exceeding 1,000 words. This justifies the use of a text chunking strategy to ensure the LLM receives relevant context without exceeding token limits.

### Data Preprocessing

To improve embedding quality, all narratives underwent:

1. **Lowercasing:** Standardizing text for consistent vectorization.
2. **Boilerplate Removal:** Stripping phrases like "I am writing to file a complaint" to reduce noise.
3. **Cleaning:** Removing special characters and extra whitespace.

---

## 🔍 Task 2: Vector Search Infrastructure

### Sampling Strategy

To ensure efficiency on local hardware, I implemented a **stratified sampling strategy**. I selected **14,825 complaints** proportionally across the five product categories to prevent bias toward high-volume categories like Credit Cards.

### Text Chunking & Embedding

Long narratives were broken into manageable pieces to optimize semantic search:

* **Strategy:** `RecursiveCharacterTextSplitter`.
* **Chunk Size:** 500 characters.
* **Chunk Overlap:** 50 characters.
* **Total Chunks Created:** 40,991.

### Vector Store & Metadata

* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` chosen for its efficiency (384 dimensions) and performance on CPU.
* **Vector Database:** **FAISS** index persisted in `vector_store/faiss_index`.
* **Traceability:** Each vector stores metadata (Original Complaint ID and Product Category) to ensure retrieved chunks can be traced back to their source.

---

## 🤖 Task 3: RAG Core Logic & Evaluation

### RAG Pipeline Implementation

The core engine of CrediTrust Intelligent Complaint Analysis is now fully functional. It utilizes a **Retrieve-Augment-Generate** architecture:

1. **Retrieval:** Uses `all-MiniLM-L6-v2` to find the top 5 most relevant complaint chunks from the FAISS vector store.
2. **Augmentation:** A custom "Senior Financial Analyst" prompt template is used to ground the LLM's response in the retrieved data, preventing hallucinations.
3. **Generation:** Powered by a quantized **Zephyr-7B-Beta** model (GGUF format), optimized for high-speed inference on standard CPU hardware.

### Hardware Optimization & Local LLM

To accommodate hardware constraints (CPU-only and 15GB disk limitations), the pipeline was pivoted from standard Transformers to the `ctransformers` library.

* **Model:** `zephyr-7b-beta.Q4_K_M.gguf` (approx. 4.4GB).
* **Execution:** Runs entirely locally on the `D:` drive via `HF_HOME` configuration.
* **Performance:** Achieves real-time response generation without requiring an expensive GPU or active internet connection.

### Qualitative Evaluation

The system was tested against 5 representative financial queries. The evaluation focused on **groundedness** (strictly using provided context) and **traceability** (citing source IDs).

| Question Category | Accuracy | Source Cite | Quality Score (1-5) |
| --- | --- | --- | --- |
| Hidden Fees | High | Yes | 5 |
| Identity Theft | High | Yes | 5 |
| Account Transfers | Medium-High | Yes | 4 |
| Customer Service | High | Yes | 5 |
| Interest Rates | Medium-High | Yes | 4 |

---

## 💻 Task 4: Interactive Analyst Interface

### Web Interface Implementation
To make the RAG system accessible to non-technical users, a web-based chat interface was built using **Gradio**. This interface allows product managers to query the complaint database without writing code.

* **Framework:** Gradio (v4.44.1 Stable).
* **Core Features:** * **Streaming Output:** Responses are rendered token-by-token, providing immediate feedback even while the CPU is processing.
    * **Trust & Verification:** The interface explicitly displays the **Verified Source Narratives** and their original **Complaint IDs** below every answer.
    * **State Management:** Includes a "Clear" button to reset the session.

### Technical Challenges & Solutions
* **Environment Stability:** Encountered breaking changes in Gradio 6.0 regarding dictionary-based message formats. Solved by downgrading to a stable 4.x version and implementing a "Fail-Safe" Interface mode.
* **Performance Optimization:** Given the 2-minute inference time on CPU for a 7B parameter model, I implemented a queuing system (`demo.queue()`) to handle long-running generation tasks without timing out the browser session.

### How to Run the App
1. Ensure the `Zephyr-7B` GGUF model is in your `D:/models/` directory.
2. Activate your environment and run:
   ```bash
   python app.py
   ```

3. Open the local URL provided (typically `http://127.0.0.1:7860`).

---

## 🏆 Project Conclusion

The CrediTrust Intelligent Complaint Analysis tool successfully demonstrates how open-source LLMs can be used to perform secure, private, and traceable analysis on sensitive financial data using local hardware.

---