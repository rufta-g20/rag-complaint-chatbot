import os
import logging
from typing import List, Tuple, Dict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_local_vector_db(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector store not found at {path}.")
    
    logger.info(f"Loading Vector Store from: {path}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def get_retriever_context(query: str, vector_db, k: int = 5) -> Tuple[str, List]:
    docs = vector_db.similarity_search(query, k=k)
    context_parts = []
    for i, d in enumerate(docs):
        cid = d.metadata.get('complaint_id', 'N/A')
        context_parts.append(f"[Source {i+1} | ID: {cid}]\n{d.page_content}")
    return "\n\n".join(context_parts), docs

def initialize_generator(model_path="D:/models/zephyr-7b-beta.Q4_K_M.gguf"):
    """
    Loads the model from a local file on your D: drive.
    No internet required once the file is downloaded.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please download it manually first.")
        
    logger.info(f"Loading Local GGUF Model from: {model_path}")
    
    llm = CTransformers(
        model=model_path, # Points directly to your local file
        model_type="mistral",
        config={
            'max_new_tokens': 300,
            'temperature': 0.7,
            'context_length': 2048
        }
    )
    return llm

def run_rag_pipeline(query: str, vector_db, llm) -> Tuple[str, List]:
    # 1. RETRIEVAL
    context, raw_docs = get_retriever_context(query, vector_db)
    
    # 2. PROMPT ENGINEERING (Optimized for Zephyr GGUF)
    prompt = f"""<|system|>
You are a Senior Financial Analyst for CrediTrust. Answer the user's question using ONLY the context provided.
Always cite the Source IDs from the context. If the answer isn't there, say you don't know.</s>
<|user|>
Context:
{context}

Question: {query}</s>
<|assistant|>"""

    # 3. GENERATION
    answer = llm.invoke(prompt)
    
    return answer.strip(), raw_docs