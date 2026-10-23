"""
BookWise AI — Web Application Interface.
Find answers in your documents. Verify every source.

Strict visual identity: Light, warm ivory, soft cream, sage green, and terracotta.
NO black or blue backgrounds anywhere in the product.
"""

import time
import streamlit as st
import pymupdf
from pathlib import Path

from config import settings
from src.bookwise.vector_store import VectorStore
from src.bookwise.retrieval import Retriever
from src.bookwise.answer_engine import AnswerEngine, NOT_FOUND_FALLBACK
from src.bookwise.evaluation import EvaluationRunner, FIVE_REQUIRED_QUESTIONS

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="BookWise AI — Document Intelligence",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Strict Light Warm Design System (CSS)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Background and Typography */
    .stApp {
        background-color: #faf8f5 !important;
        color: #2b2826 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f5f0e6 !important;
        border-right: 1px solid #e8e1d7 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #2b2826 !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2b2826 !important;
        font-weight: 600 !important;
    }
    
    /* Primary Buttons (Terracotta / Sage) */
    .stButton > button {
        background-color: #c86446 !important;
        color: #ffffff !important;
        border: 1px solid #b35338 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #b35338 !important;
        border-color: #9c442c !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(200, 100, 70, 0.2) !important;
    }
    .stButton > button:active {
        background-color: #9c442c !important;
    }

    /* Cards and Content Panels */
    .bw-card {
        background-color: #ffffff;
        border: 1px solid #e8e1d7;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 8px rgba(43, 40, 38, 0.04);
    }
    
    .bw-card-subtle {
        background-color: #fbf9f6;
        border: 1px solid #f0ebe1;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }

    /* Badges */
    .bw-badge-sage {
        background-color: #eaf0e8;
        color: #3b5036;
        border: 1px solid #c8d8c4;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.825rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .bw-badge-terracotta {
        background-color: #faece8;
        color: #8c432d;
        border: 1px solid #f0cec4;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.825rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .bw-badge-neutral {
        background-color: #ede7dc;
        color: #5a534c;
        border: 1px solid #dcd4c7;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.825rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Evidence Block */
    .bw-evidence {
        border-left: 4px solid #c86446;
        background-color: #fbf9f6;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        font-style: italic;
        color: #4a423b;
        margin-top: 0.75rem;
    }

    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #2b2826 !important;
        border: 1px solid #dcd4c7 !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #c86446 !important;
        box-shadow: 0 0 0 2px rgba(200, 100, 70, 0.2) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        gap: 1rem !important;
        border-bottom: 1px solid #e8e1d7 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #5a534c !important;
        font-weight: 500 !important;
        padding: 0.6rem 1rem !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #c86446 !important;
        border-bottom: 2px solid #c86446 !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine():
    """Cache singleton AnswerEngine instance."""
    return AnswerEngine()


@st.cache_resource
def get_vector_store():
    """Cache singleton VectorStore instance."""
    return VectorStore()


engine = get_engine()
vector_store = get_vector_store()
stats = vector_store.get_stats()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📖 BookWise AI")
    st.markdown("*Find answers in your documents. Verify every source.*")
    st.markdown("---")

    st.markdown("#### Document Status")
    st.markdown(f"**Document**: `{stats['source_filename']}`")
    st.markdown(f"**Index Status**: <span class='bw-badge-sage'>READY ({stats['total_chunks']} chunks)</span>", unsafe_allow_html=True)
    st.markdown(f"**Total Pages**: `91 pages`")
    st.markdown(f"**Collection**: `{stats['collection_name']}`")
    st.markdown(f"**Embedding**: `MiniLM-L6-v2 (Local ONNX)`")
    st.markdown(f"**Engine**: `Deterministic Grounded RAG`")
    st.markdown("---")

    st.markdown("#### Core Product Principles")
    st.markdown("1. **Evidence before answers**")
    st.markdown("2. **Accurate citations before confidence**")
    st.markdown("3. **Explicit abstention (`Not found`)**")
    st.markdown("4. **Zero black or blue styling**")
    st.markdown("---")

    st.caption("BookWise AI v1.0.0 — Commercial Document Intelligence MVP")

# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------
st.markdown(
    """
    <div style='padding-bottom: 1rem;'>
        <h1 style='margin-bottom: 0.25rem;'>BookWise AI</h1>
        <p style='color: #5a534c; font-size: 1.15rem; margin-top: 0;'>
            Evidence-grounded Document Intelligence & Source Verification
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_qa, tab_inspector, tab_benchmark, tab_arch = st.tabs([
    "💬 Document Q&A",
    "🔍 Source Page Inspector",
    "📊 Benchmark Evaluation",
    "🏛 Architecture & Flow"
])

# ---------------------------------------------------------
# TAB 1: Document Q&A
# ---------------------------------------------------------
with tab_qa:
    st.markdown("#### Ask a question about the Orient BlackSwan History Catalogue")
    st.caption("Every factual answer is verified against indexed source pages. Unsupported questions return exactly **'Not found'**.")

    # Quick test pills
    st.markdown("**Try acceptance questions:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Session State Initialization
    if "active_query" not in st.session_state:
        st.session_state["active_query"] = "On what date were prices in the catalogue last updated?"
    if "run_search" not in st.session_state:
        st.session_state["run_search"] = True
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    def select_preset(preset_text: str):
        st.session_state["active_query"] = preset_text
        st.session_state["run_search"] = True

    # Quick test preset pills
    st.markdown("**Sample Acceptance Questions (Click to test):**")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.button("📅 Price Date", on_click=select_preset, args=("On what date were prices in the catalogue last updated?",), use_container_width=True)
    with col2:
        st.button("🏢 Incorporation", on_click=select_preset, args=("In which year was Orient BlackSwan incorporated?",), use_container_width=True)
    with col3:
        st.button("🔬 Physics Price", on_click=select_preset, args=("What is the price of a Physics textbook?",), use_container_width=True)
    with col4:
        st.button("🏛 Minister Query", on_click=select_preset, args=("Who is the current Minister of Education?",), use_container_width=True)
    with col5:
        st.button("⭐ 'Best' Book", on_click=select_preset, args=('Which is the "best" history book for a beginner?',), use_container_width=True)

    # Search Form
    with st.form(key="search_form", clear_on_submit=False):
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            user_query = st.text_input(
                "Enter your question:",
                value=st.session_state["active_query"],
                placeholder="Ask any question about the catalogue...",
            )
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button("Ask BookWise AI", type="primary", use_container_width=True)

    # Execute search when form submitted or pill clicked
    if submit:
        st.session_state["active_query"] = user_query
        st.session_state["run_search"] = True

    if st.session_state["run_search"] and st.session_state["active_query"]:
        query_to_run = st.session_state["active_query"]
        with st.spinner("Retrieving evidence and verifying citations..."):
            st.session_state["last_result"] = engine.ask(query_to_run)
        st.session_state["run_search"] = False

    # Display Answer Result
    res = st.session_state.get("last_result")
    if res:
        st.markdown("---")
        st.markdown("### Answer")

        if res.is_supported:
            st.markdown(
                f"""
                <div class='bw-card'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;'>
                        <span class='bw-badge-sage'>✓ EVIDENCE SUPPORTED</span>
                        <span style='color: #857c74; font-size: 0.85rem;'>Latency: {res.execution_time_ms:.1f} ms</span>
                    </div>
                    <div style='font-size: 1.35rem; font-weight: 600; color: #2b2826; margin-bottom: 1rem;'>
                        {res.answer}
                    </div>
                    <div style='margin-bottom: 0.75rem;'>
                        <strong style='color: #5a534c;'>Verified Citation:</strong>
                        <span class='bw-badge-terracotta' style='margin-left: 0.5rem;'>
                            {res.citations[0].format_citation() if res.citations else 'None'}
                        </span>
                    </div>
                    <div>
                        <strong style='color: #5a534c;'>Direct Source Evidence:</strong>
                        <div class='bw-evidence'>
                            "{res.evidence_excerpts[0]}"
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class='bw-card'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;'>
                        <span class='bw-badge-neutral'>⊘ ABSTENTION (NOT FOUND)</span>
                        <span style='color: #857c74; font-size: 0.85rem;'>Latency: {res.execution_time_ms:.1f} ms</span>
                    </div>
                    <div style='font-size: 1.35rem; font-weight: 600; color: #5a534c; margin-bottom: 0.5rem;'>
                        {res.answer}
                    </div>
                    <p style='color: #857c74; font-size: 0.9rem; margin: 0;'>
                        <strong>Reason:</strong> {res.reason}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Candidate Inspection Expander
        with st.expander("Inspect Retrieved Evidence Candidates"):
            st.markdown(f"**Retrieved Candidates:** {len(res.retrieved_candidates)}")
            for i, cand in enumerate(res.retrieved_candidates):
                st.markdown(
                    f"""
                    <div class='bw-card-subtle'>
                        <strong>Candidate #{i+1}</strong> — PDF Page {cand.pdf_page} ({cand.printed_page_str})
                        <br><span style='color: #857c74; font-size: 0.85rem;'>Score: {cand.similarity_score:.3f} | Distance: {cand.distance:.3f}</span>
                        <div style='margin-top: 0.5rem; font-size: 0.9rem; color: #4a423b; white-space: pre-wrap;'>{cand.text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------
# TAB 2: Source Page Inspector
# ---------------------------------------------------------
with tab_inspector:
    st.markdown("#### Direct PDF Source Page Inspector")
    st.caption("Inspect the authentic extracted text for any page of History_2024-25.pdf.")

    col_page, col_info = st.columns([1, 3])
    with col_page:
        selected_page = st.number_input("PDF Page Number (1 to 91):", min_value=1, max_value=91, value=2)

    doc_path = settings.pdf_path
    if doc_path.exists():
        doc = pymupdf.open(str(doc_path))
        page_idx = selected_page - 1
        page_obj = doc[page_idx]
        page_text = page_obj.get_text()
        doc.close()

        st.markdown(
            f"""
            <div class='bw-card'>
                <h4>PDF Page {selected_page} of 91</h4>
                <div style='background-color: #fbf9f6; padding: 1.25rem; border-radius: 8px; font-family: monospace; font-size: 0.875rem; white-space: pre-wrap; color: #2b2826; border: 1px solid #e8e1d7; max-height: 500px; overflow-y: auto;'>
{page_text if page_text.strip() else '[Cover image / No selectable text on this page]'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# TAB 3: Benchmark Evaluation
# ---------------------------------------------------------
with tab_benchmark:
    st.markdown("#### Automated Acceptance Benchmark Suite")
    st.caption("Verifies the 5 required questions and evaluates accuracy, citation validity, and abstention precision.")

    if st.button("Run Live Benchmark", type="primary"):
        eval_runner = EvaluationRunner(engine)
        with st.spinner("Executing benchmark across test cases..."):
            metrics = eval_runner.run_eval()

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Answer Accuracy", f"{metrics['answer_accuracy']}%")
        m_col2.metric("Citation Accuracy", f"{metrics['citation_accuracy']}%")
        m_col3.metric("Abstention Precision", f"{metrics['abstention_accuracy']}%")
        m_col4.metric("Avg Latency", f"{metrics['average_latency_ms']} ms")

        st.markdown("##### Detailed Question Results:")
        for r in metrics["results"]:
            badge = "<span class='bw-badge-sage'>PASS</span>" if r["is_correct"] and r["citation_valid"] else "<span class='bw-badge-terracotta'>FAIL</span>"
            st.markdown(
                f"""
                <div class='bw-card-subtle' style='margin-bottom: 0.5rem;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <strong>{r['id']}: {r['question']}</strong>
                        {badge}
                    </div>
                    <div style='font-size: 0.9rem; margin-top: 0.25rem;'>
                        <strong>Expected:</strong> {r['expected']} | <strong>Actual:</strong> {r['actual']}
                        <br><strong>Citation:</strong> {r['citation']} | <strong>Latency:</strong> {r['latency_ms']} ms
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------
# TAB 4: Architecture
# ---------------------------------------------------------
with tab_arch:
    st.markdown("#### BookWise AI Architecture & Pipeline")
    st.markdown(
        """
        ```
        Source PDF (History_2024-25.pdf)
            ↓ [PyMuPDF Page-by-Page Extraction & Normalization]
        DocumentPages (91 pages with PDF Page & Printed Page metadata)
            ↓ [LangChain RecursiveCharacterTextSplitter]
        TextChunks (895 cohesive chunks with content SHA-256 hashes)
            ↓ [Local ONNX all-MiniLM-L6-v2 Embeddings]
        ChromaDB Vector Store (Persistent cosine index with manifest)
            ↓ [Hybrid Candidate Retrieval + Neighbor Expansion]
        Retrieved Candidates (Top-K semantic + entity boost + adjacent chunks)
            ↓ [Evidence Support Validation]
        Verified Grounding Decision
            ├── Supported → Verified Answer + Source Citation
            └── Unsupported / Subjective → Exact 'Not found'
        ```
        """
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #857c74; font-size: 0.85rem; padding: 1rem 0;'>
        BookWise AI — Answers are strictly grounded in indexed document evidence. Users can inspect all cited source pages.
    </div>
    """,
    unsafe_allow_html=True,
)
