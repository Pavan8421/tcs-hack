import streamlit as st
import logging
import os
import io
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import faiss
import pickle
import json
import numpy as np
import re

# --- In-memory log capture for Streamlit ---
log_stream = io.StringIO()
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# --- File-based log capture ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "chatbot_logs.log")

file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# --- Configure logger ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Avoid adding handlers multiple times if Streamlit reruns
if not logger.hasHandlers():
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

logger.propagate = False

# --- Load Models Safely ---
@st.cache_resource
def load_models():
    try:
        logger.info("🔄 Loading FAISS index and metadata...")
        faiss_index = faiss.read_index("faiss_index_bge_base.index")
        with open("faiss_metadata.pkl", "rb") as f:
            passages = pickle.load(f)
        logger.info(f"✅ Loaded {len(passages)} passages.")

        logger.info("🔄 Loading embedding model on CPU...")
        embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        logger.info("✅ Embedding model loaded.")

        logger.info("🔄 Loading Mistral model...")
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        logger.info("✅ Mistral model + tokenizer loaded.")


        # Load Zephyr for enrichment (as pipeline)
        logger.info("🔄 Loading Zephyr-7B chat pipeline for offline chunk enrichment...")
        pipe = pipeline(
            "text-generation",
            model="HuggingFaceH4/zephyr-7b-beta",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        logger.info("✅ Zephyr chat pipeline loaded.")

        return faiss_index, passages, embedding_model, model, tokenizer, pipe

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# --- Load everything ---
faiss_index, passages, embedding_model, model, tokenizer, pipe = load_models()

# 🔍 Safety check
if passages is None:
    st.error("❌ 'passages' is None. Check your faiss_metadata.pkl file.")
    st.stop()

if not isinstance(passages, list):
    st.error("❌ 'passages' must be a list. Got: " + str(type(passages)))
    st.stop()

if len(passages) == 0:
    st.warning("⚠️ 'passages' is empty.")


# --- PDF Upload Section (Top Right Sidebar) ---
with st.sidebar:
    st.header("➕ Add New PDF")
    
    uploaded_file = st.file_uploader("Upload an insurance policy PDF", type=["pdf"], key="pdf_uploader")

    # Initialize session state variables
    if "upload_processed" not in st.session_state:
        st.session_state.upload_processed = False
    if "last_uploaded_filename" not in st.session_state:
        st.session_state.last_uploaded_filename = ""

    # If a new file is uploaded, reset flags
    if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_filename:
        st.session_state.upload_processed = False
        st.session_state.last_uploaded_filename = uploaded_file.name

    if uploaded_file and not st.session_state.upload_processed:
        logger.info("📤 Uploaded file received and not yet processed.")
        
        os.makedirs("policy_pdfs", exist_ok=True)
        pdf_save_path = os.path.join("policy_pdfs", uploaded_file.name)

        # Save the file
        file_content = uploaded_file.read()
        with open(pdf_save_path, "wb") as f:
            f.write(file_content)

        st.success(f"✅ File saved as: {uploaded_file.name}")
        logger.info(f"✅ Saved uploaded file to: {pdf_save_path}")

        # Run the pipeline
        from pathlib import Path
        import time
        from knowledge_update import (
            extract_text_from_pdf,
            chunk_text_by_paragraphs,
            enrich_chunk_with_zephyr
        )

        st.info("🔄 Extracting and enriching...")
        text = extract_text_from_pdf(pdf_save_path)
        chunks = chunk_text_by_paragraphs(text)

        enriched_chunks = []
        for chunk in chunks:
            enriched_chunk = enrich_chunk_with_zephyr(
                section_text=chunk["content"],
                section_title=chunk["section_title"],
                source=uploaded_file.name,
                pipe=pipe
            )
            enriched_chunks.append(enriched_chunk)

        enriched_json_path = f"./enriched_jsons/{Path(uploaded_file.name).stem}_enriched.json"
        with open(enriched_json_path, "w", encoding="utf-8") as f:
            json.dump(enriched_chunks, f, indent=2, ensure_ascii=False)

        st.info("🧠 Updating FAISS index...")
        for chunk in enriched_chunks:
            text = chunk["text"]
            metadata = chunk["metadata"]
            metadata.pop("error", None)
            vector = embedding_model.encode(text, show_progress_bar=False)
            faiss_index.add(np.array([vector]).astype("float32"))
            passages.append(metadata)

        # Save updated FAISS + metadata
        faiss.write_index(faiss_index, "faiss_index_bge_base.index")
        with open("faiss_metadata.pkl", "wb") as f:
            pickle.dump(passages, f)

        st.success("✅ PDF indexed and chatbot updated!")

        # ✅ Mark as processed and rerun to reset uploader
        st.session_state.upload_processed = True
        time.sleep(1)
        st.experimental_rerun()



st.title("🛡️ Insurance Policy Chatbot (Mistral 7B + Hybrid FAISS)")
# ✅ Debug structure of one passage
#st.markdown("### 🧪 Sample Passage")
#st.write("Sample passage:", passages[0])
# --- Utility ---
def clean_text(text):
    return re.sub(r"\W+", " ", text.lower()).split()

def hybrid_search(query, top_k=5, alpha=0.6):
    logger.info(f"🔍 Running hybrid search for: {query}")
    query_vec = embedding_model.encode(query)
    dense_scores, dense_indices = faiss_index.search(query_vec.reshape(1, -1), len(passages))

    query_words = set(clean_text(query))
    keyword_scores = []

    for idx in dense_indices[0]:
        doc = passages[idx]
        meta = doc
        section_text = meta.get("section_title", "") + " " + meta.get("content", "")
        doc_words = set(clean_text(section_text))
        overlap = len(query_words & doc_words)
        keyword_scores.append((idx, overlap))

    dense_scores = dense_scores[0]
    dense_scores = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores) + 1e-6)
    keyword_vals = np.array([score for _, score in keyword_scores])
    keyword_vals = (keyword_vals - np.min(keyword_vals)) / (np.max(keyword_vals) - np.min(keyword_vals) + 1e-6)

    combined_scores = alpha * dense_scores + (1 - alpha) * keyword_vals
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        doc = passages[dense_indices[0][i]]
        meta = doc
        # results.append(meta)
        results.append({
            "content": meta.get("content", ""),
            "policy_type": meta.get("policy_type", ""),
            "coverage": meta.get("coverage", "")
        })

    logger.info(f"✅ Retrieved {len(results)} top sections for query: {query}")
    return results


def build_prompt(context: str, query: str) -> str:
    return f"""
You are a helpful and knowledgeable assistant specialized in insurance policies. Based on the provided context, respond clearly and politely to the user's question. Your answer should be accurate, easy to understand, and based only on the given context.

If you cannot find a reliable answer from the context, simply respond with:
"Sorry to say, I don’t know the answer for this question."

Context:
{context}

User Question:
{query}

Chatbot Answer:"""

# --- UI Input ---
user_input = st.text_input("Ask your question about the policy:")

import re

if user_input:
    try:
        context = hybrid_search(user_input)
        prompt = build_prompt(context, user_input)

        with st.spinner("Thinking..."):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(inputs.input_ids, max_new_tokens=256, do_sample=False)
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ✅ Extract only the part after "Chatbot Answer:"
        match = re.search(r"Chatbot Answer:\s*(.*)", full_response, re.DOTALL)
        response = match.group(1).strip() if match else full_response.strip()

        st.markdown("### 🧠 Response:")
        st.write(response)

    except Exception as e:
        st.error("⚠️ Something went wrong while generating a response.")
        logger.error(f"Response error: {e}")

# --- Show Logs in UI ---
#st.markdown("### 📜 Debug Logs")
#st.code(log_stream.getvalue())
