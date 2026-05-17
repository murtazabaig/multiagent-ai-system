import streamlit as st
import json
import logging
import sys
import time
import io
import PyPDF2

# Ensure stdout uses utf-8 in the background to prevent encoding errors
from src.core.pipeline import Pipeline

st.set_page_config(page_title="Agenta | Neural AI QA", page_icon="🪐", layout="wide", initial_sidebar_state="expanded")

# --- PREMIUM CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Glassmorphism background for the main area */
    .stApp {
        background: radial-gradient(circle at 15% 50%, rgba(15, 23, 42, 1) 0%, rgba(2, 6, 23, 1) 100%);
        color: #e2e8f0;
    }
    
    /* Headers */
    h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        letter-spacing: -1.5px;
        margin-bottom: 0.5rem;
    }
    h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
        font-weight: 600;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.25rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #f8fafc;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    .stTextInput > div > div > input:focus {
        border-color: #818cf8;
        box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.2);
        background-color: rgba(30, 41, 59, 0.8);
    }
    
    /* Modern Button */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5);
        color: white;
    }
    .stButton>button:active {
        transform: translateY(1px);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(129, 140, 248, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(79, 70, 229, 0.2);
        color: #818cf8;
        border-bottom: 2px solid #818cf8;
    }
    
    /* Success Message */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
    }
    
    .final-answer {
        font-size: 1.5rem;
        font-weight: 600;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Caches the pipeline so it doesn't reload the HuggingFace models 
    or the FAISS index on every single UI interaction.
    """
    return Pipeline()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8633/8633190.png", width=60)
    st.markdown("## ⚙️ Control Center")
    st.markdown("Configure the Multi-Agent neural pathways.")
    
    st.slider("Retrieval Top-K", min_value=1, max_value=10, value=5, help="Number of context segments to retrieve per subtask.")
    st.selectbox("LLM Reasoner", ["google/flan-t5-small", "google/flan-t5-base", "meta-llama/Llama-3-8B"], index=0, disabled=True)
    st.selectbox("Vector Store", ["FAISS (19k SQuAD Vectors)"], disabled=True)
    
    st.markdown("---")
    st.markdown("### 📊 System Architecture")
    st.markdown("🔹 **Planner Agent:** `flan-t5-base`")
    st.markdown("🔹 **Retriever Agent:** `MiniLM-L6-v2`")
    st.markdown("🔹 **Executor Agent:** `flan-t5-base`")
    st.markdown("🔹 **Embeddings:** FAISS HNSW")

    st.markdown("---")
    st.markdown("### 📄 Custom Knowledge Base")
    st.markdown("Upload your own PDFs to inject new facts into the AI's vector database.")
    uploaded_file = st.file_uploader("Upload PDF / Document", type=["pdf", "txt"])
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Extracting & Vectorizing..."):
                try:
                    text = ""
                    if uploaded_file.name.endswith(".pdf"):
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                        for page in pdf_reader.pages:
                            text += page.extract_text() + " "
                    else:
                        text = uploaded_file.read().decode("utf-8")
                    
                    # Basic Chunking (approx 500 chars per chunk)
                    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                    
                    pipeline = load_pipeline()
                    pipeline.add_documents(chunks)
                    st.success(f"Successfully vectorized {len(chunks)} chunks into FAISS!")
                except Exception as e:
                    st.error(f"Failed to ingest: {e}")

# --- MAIN CONTENT ---
st.title("🪐 Neural Multi-Agent System")
st.markdown('<p class="subtitle">Autonomous reasoning pipeline synthesizing insights across 19,029 verified knowledge vectors.</p>', unsafe_allow_html=True)

with st.spinner("⏳ Loading Neural Pathways & Vector Index (This may take up to 60 seconds on first run)..."):
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Failed to load pipeline: {str(e)}")
        st.stop()

# Input area
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🎯 Mission Objective")
query = st.text_input("Query", "What is the capital of France and why is it important?", placeholder="Ask the Multi-Agent system anything...", label_visibility="collapsed")
submit_button = st.button("Initialize Sequence")
st.markdown('</div>', unsafe_allow_html=True)

if submit_button:
    st.markdown("---")
    
    start_time = time.time()
    
    with st.spinner("⚡ Agents are orchestrating a response..."):
        try:
            result = pipeline.run(query)
            end_time = time.time()
            elapsed = end_time - start_time
            
            # --- FINAL ANSWER ---
            st.markdown("### ✨ Executive Synthesis")
            final_execution = result["results"][-1] if "results" in result and result["results"] else {}
            if "result" in final_execution:
                # Use st.success directly without raw HTML to avoid literal tag rendering
                st.success(f"**{final_execution['result']}**", icon="✅")
            else:
                st.error("The reasoning agents failed to synthesize a final conclusion.")
            
            st.caption(f"⏱️ Sequence completed in {elapsed:.2f} seconds")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- WORKFLOW BREAKDOWN ---
            st.markdown("### 🔬 Autonomous Workflow Telemetry")
            
            tab1, tab2, tab3 = st.tabs(["📋 1. Planner Agent", "📚 2. Retrieval Space", "⚙️ 3. Execution Engine"])
            
            with tab1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Task Deconstruction Matrix")
                if "plan" in result and "subtasks" in result["plan"]:
                    st.json(result["plan"]["subtasks"])
                else:
                    st.info("No subtasks generated.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with tab2:
                for idx, res in enumerate(result.get("results", [])):
                    if "documents" in res and res["documents"]:
                        st.markdown(f"**Step {idx + 1} Retrieval (Found {len(res['documents'])} verified sources):**")
                        for i, (doc, score) in enumerate(zip(res["documents"], res.get("scores", [0]*len(res["documents"])))):
                            with st.expander(f"📄 Source #{i+1} (Vector Confidence: {score:.4f})"):
                                st.markdown(f"<p style='color: #cbd5e1; line-height: 1.6;'>{doc}</p>", unsafe_allow_html=True)
                        st.markdown("<hr>", unsafe_allow_html=True)
                
            with tab3:
                for idx, res in enumerate(result.get("results", [])):
                    if "reasoning" in res:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown(f"#### Cycle {idx + 1} Internal Monologue")
                        st.info(f"**Thoughts:**\n\n{res['reasoning']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"⚠️ Critical Sequence Failure: {str(e)}")

