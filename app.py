import gradio as gr
import time
from src.retrieval_pipeline import load_local_vector_db, initialize_generator, run_rag_pipeline

# 1. Configuration & Loading
VECTOR_DB_PATH = "vector_store/faiss_index"
MODEL_PATH = "D:/models/zephyr-7b-beta.Q4_K_M.gguf"

print("Initializing CrediTrust Analyst Interface (Gradio 4.44.1 Stable)...")
db = load_local_vector_db(VECTOR_DB_PATH)
llm = initialize_generator(MODEL_PATH)

def ask_analyst(query):
    """
    Standard function for Gradio 4.x Interface.
    """
    if not query or not query.strip():
        return "Please enter a question."

    try:
        # 1. Run RAG logic
        answer, sources = run_rag_pipeline(query, db, llm)
        
        # 2. Format Source Display
        source_display = "\n\n" + "-"*30 + "\n### 📄 Verified Sources (Metadata):\n"
        for i, doc in enumerate(sources[:3]):
            cid = doc.metadata.get('complaint_id', 'N/A')
            source_display += f"**Source {i+1} [ID: {cid}]:** {doc.page_content[:200]}...\n\n"
        
        full_response = f"{answer}\n{source_display}"
        
        # 3. Yielding for streaming effect
        output = ""
        for char in full_response:
            output += char
            time.sleep(0.002) # Fast streaming
            yield output

    except Exception as e:
        yield f"⚠️ System Error: {str(e)}"

# 2. Building the UI with 4.x parameters
demo = gr.Interface(
    fn=ask_analyst,
    inputs=gr.Textbox(lines=2, placeholder="Ask about customer complaints...", label="Your Question"),
    outputs=gr.Markdown(label="CrediTrust Analyst Response"),
    title="🏦 CrediTrust Intelligent Analyst",
    description="Query customer complaint narratives using local RAG technology. (V4.44.1 Stable)",
    theme="soft",
    allow_flagging="never"  # This is the correct placement for 4.x
)

if __name__ == "__main__":
    # Ensure queue is enabled for streaming (yield) to work
    demo.queue().launch()