import pandas as pd
import os
import logging
import argparse
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Setup logging for better monitoring and auditability
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the cleaned dataset generated in Task 1.
    
    Args:
        file_path (str): Path to the processed CSV file.
    Returns:
        pd.DataFrame: Loaded complaint data.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed data not found at {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def get_stratified_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Creates a stratified sample to ensure proportional representation across product categories.
    
    Args:
        df (pd.DataFrame): The full cleaned dataset.
        sample_size (int): Total desired number of samples.
    Returns:
        pd.DataFrame: The sampled dataset.
    """
    n_products = df['Product'].nunique()
    samples_per_group = sample_size // n_products
    
    # Stratified sampling ensures Asha gets balanced insights across all CrediTrust products
    df_sample = df.groupby('Product', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), samples_per_group), random_state=42)
    ).reset_index(drop=True)
    
    logging.info(f"Stratified sampling complete. Total sample size: {len(df_sample)}")
    logging.info(f"Samples per product category: ~{samples_per_group}")
    return df_sample

def build_documents(df: pd.DataFrame, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Implements a text chunking strategy to handle long narratives effectively.
    
    Args:
        df (pd.DataFrame): Sampled dataframe.
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Overlap between chunks to preserve context.
    Returns:
        List[Document]: List of LangChain Document objects with metadata.
    """
    # RecursiveCharacterTextSplitter is chosen to keep related sentences/paragraphs together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents = []
    chunk_counts = []

    for _, row in df.iterrows():
        narrative = str(row['cleaned_narrative'])
        chunks = text_splitter.split_text(narrative)
        chunk_counts.append(len(chunks))
        
        for i, chunk in enumerate(chunks):
            # Metadata is crucial for traceability back to the original CFPB record
            doc = Document(
                page_content=chunk,
                metadata={
                    "complaint_id": str(row.get('Complaint ID', 'N/A')),
                    "product": str(row.get('Product', 'N/A')),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
    
    # Logging key statistics for monitoring as requested by reviewers
    logging.info(f"Chunking complete: {len(documents)} total chunks created.")
    logging.info(f"Average chunks per complaint: {sum(chunk_counts)/len(chunk_counts):.2f}")
    return documents

def build_vector_store(documents: List[Document], model_name: str) -> FAISS:
    """
    Generates embeddings and builds the FAISS vector index.
    
    Args:
        documents (List[Document]): Chunks to embed.
        model_name (str): HuggingFace model path.
    Returns:
        FAISS: The initialized vector store.
    """
    logging.info(f"Initializing embedding model: {model_name}")
    try:
        # all-MiniLM-L6-v2 is used for its balance of speed and semantic accuracy
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        logging.info("Generating embeddings and building FAISS index...")
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        logging.error(f"Critical error during vector store creation: {e}")
        raise

def parse_args():
    """Parameterize execution via CLI arguments."""
    parser = argparse.ArgumentParser(description="Task 2: Chunking, Embedding, and Indexing")
    parser.add_argument("--data_path", type=str, default="data/processed/filtered_complaints.csv")
    parser.add_argument("--index_path", type=str, default="vector_store/faiss_index")
    parser.add_argument("--sample_size", type=int, default=15000)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    try:
        # 1. Load data from Task 1 output
        df_processed = load_data(args.data_path)

        # 2. Perform Stratified Sampling
        df_sample = get_stratified_sample(df_processed, args.sample_size)

        # 3. Chunk text and generate Document objects
        docs = build_documents(df_sample, args.chunk_size, args.chunk_overlap)

        # 4. Generate Embeddings and Index
        vs = build_vector_store(docs, args.model)

        # 5. Persist the Vector Store
        os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
        vs.save_local(args.index_path)
        
        logging.info(f"Task 2 Successful! Vector store persisted at: {args.index_path}")

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")