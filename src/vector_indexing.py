import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Path Setup ---
# Ensures the script finds the data folder regardless of where you run it from
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
file_path = os.path.join(project_root, 'data', 'processed', 'filtered_complaints.csv')

print(f"Loading cleaned data from: {file_path}")

# 1. Load the cleaned dataset from Task 1
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Could not find {file_path}. Please run Task 1 first.")
    exit()

# 2. Stratified Sampling (10,000 - 15,000 complaints) 
# We calculate samples per group manually to avoid Pandas version-specific errors
sample_size = 15000
n_products = df['Product'].nunique()
samples_per_group = sample_size // n_products

# Use a lambda that keeps the group structure intact for the metadata later
df_sample = df.groupby('Product', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), samples_per_group), random_state=42)
).reset_index(drop=True)

print(f"Stratified sample created with {len(df_sample)} records.")

# 3. Text Chunking 
# Strategy: Recursive splitting with overlap to preserve semantic context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

documents = []
for _, row in df_sample.iterrows():
    # Split the narrative into chunks
    narrative_text = str(row['cleaned_narrative'])
    chunks = text_splitter.split_text(narrative_text)
    
    # Create Document objects with metadata for traceability
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "complaint_id": str(row.get('Complaint ID', 'N/A')),
                "product": str(row.get('Product', 'N/A')),
                "chunk_index": i
            }
        )
        documents.append(doc)

print(f"Total chunks created: {len(documents)}")

# 4. Choose and Initialize Embedding Model 
# Using all-MiniLM-L6-v2: Fast, efficient (384 dims), and great for semantic search
print("Initializing embedding model and generating vectors (this may take a few minutes)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Create and Persist Vector Store (FAISS) 
# This converts text chunks to vectors and stores them in a local index
vector_store = FAISS.from_documents(documents, embeddings)

# Save the index to the vector_store/ directory at the project root
save_path = os.path.join(project_root, 'vector_store', 'faiss_index')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
vector_store.save_local(save_path)

print(f"Task 2 Complete: Vector store saved in {save_path}")