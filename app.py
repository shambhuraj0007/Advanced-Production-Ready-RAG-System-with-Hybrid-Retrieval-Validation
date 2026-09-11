"""
Streamlit UI for the RAG System
A user-friendly web interface for interacting with the RAG system.
"""

import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from src.rag_system import SimpleRAG

# Page configuration
st.set_page_config(
    page_title="RAG System - Intelligent Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        background-color: #e3f2fd;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
    .citation-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .validation-success {
        color: #4caf50;
        font-weight: bold;
    }
    .validation-warning {
        color: #ff9800;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_list' not in st.session_state:
    st.session_state.document_list = []


def initialize_rag_system(provider: Optional[str] = None) -> SimpleRAG:
    """Initialize the RAG system using configuration from .env and defaults."""
    kwargs = {}
    if provider:
        kwargs['provider'] = provider
    return SimpleRAG(**kwargs)


# Auto-initialize RAG system on launch
if st.session_state.rag_system is None:
    try:
        st.session_state.rag_system = initialize_rag_system()
        if st.session_state.rag_system.vectorstore is not None:
            st.session_state.documents_loaded = True
    except Exception as e:
        st.session_state.init_error = str(e)


def load_documents_from_directory(directory: str) -> List[str]:
    """Get all document files from a directory."""
    file_paths = []
    supported_extensions = {'.pdf', '.txt', '.md'}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in supported_extensions:
                file_paths.append(os.path.join(root, file))
    
    return file_paths


def format_citations(citations: List[Dict[str, Any]]) -> str:
    """Format citations for display."""
    if not citations:
        return "No citations available"
    
    formatted = []
    for i, citation in enumerate(citations, 1):
        c_type = citation.get('type')
        if c_type == 'web' or 'General Answer' in citation.get('file_name', ''):
            source = citation.get('source', 'Web Search')
            url = citation.get('url', '')
            if url:
                formatted.append(f"[{i}] 🌐 {source}\n    URL: {url}")
            else:
                formatted.append(f"[{i}] 🌐 {source}")
        elif c_type == 'general':
            formatted.append(f"[{i}] 🌐 General Knowledge (Not found in uploaded documents)")
        else:
            file_name = citation.get('file_name', 'Unknown')
            page = citation.get('page', 'N/A')
            path = citation.get('path', '')
            formatted.append(f"[{i}] 📄 {file_name} (Page {page})")
            if path:
                formatted.append(f"    Path: {path}")
    
    return "\n".join(formatted)


def display_validation_results(validation: Dict[str, Any]):
    """Display answer validation results."""
    if not validation:
        return
    
    is_valid = validation.get('is_valid', False)
    confidence = validation.get('confidence', 0.0)
    warnings = validation.get('warnings', [])
    supporting_evidence = validation.get('supporting_evidence', [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if is_valid:
            st.markdown(f'<p class="validation-success">✓ Valid Answer (Confidence: {confidence:.2%})</p>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="validation-warning">⚠ Validation Issues (Confidence: {confidence:.2%})</p>', 
                       unsafe_allow_html=True)
    
    with col2:
        st.metric("Confidence Score", f"{confidence:.2%}")
    
    if warnings:
        st.warning("⚠️ Warnings:")
        for warning in warnings:
            st.text(f"  • {warning}")
    
    if supporting_evidence:
        with st.expander("📚 Supporting Evidence"):
            for evidence in supporting_evidence:
                st.text(evidence)


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG System - Intelligent Document Q&A</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar for system status & quick actions
    with st.sidebar:
        st.header("⚡ System Status")
        
        if st.session_state.rag_system:
            rag = st.session_state.rag_system
            st.success(f"🟢 **Online** ({rag.provider.capitalize()})")
            
            st.markdown(f"**LLM Model:** `{rag.llm_model}`")
            st.markdown(f"**Embedding:** `{rag.embedding_model}`")
            
            features = []
            if getattr(rag, 'use_hybrid_search', True):
                features.append("Hybrid Search")
            if getattr(rag, 'use_reranking', True):
                features.append("Re-ranking")
            if getattr(rag, 'use_query_routing', True):
                features.append("Query Router")
            if getattr(rag.memory_manager, 'use_memory', True):
                features.append("Memory")
            if features:
                st.caption(f"Active features: {', '.join(features)}")
            
            st.divider()
            
            # Document status
            if st.session_state.documents_loaded:
                doc_count = len(st.session_state.document_list)
                st.info(f"📚 Documents: {doc_count if doc_count > 0 else 'Active in DB'}")
            else:
                st.warning("⚠️ No documents indexed yet. Upload files in the Documents tab.")
                
            st.divider()
            
            # Quick Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reload", use_container_width=True, help="Reload configuration from .env"):
                    try:
                        st.session_state.rag_system = initialize_rag_system()
                        st.success("Reloaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            with col2:
                if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear conversation memory"):
                    if st.session_state.rag_system:
                        st.session_state.rag_system.memory_manager.clear_memory()
                    st.session_state.chat_history = []
                    st.success("Cleared!")
                    st.rerun()
            
            # Optional collapsible provider override
            with st.expander("⚙️ Provider Switcher (.env override)", expanded=False):
                st.caption("Settings are loaded from `.env`. You can quickly switch provider here:")
                current_p = rag.provider if rag.provider in ["gemini", "groq", "openai"] else "gemini"
                selected_p = st.selectbox(
                    "Switch Provider",
                    ["gemini", "groq", "openai"],
                    index=["gemini", "groq", "openai"].index(current_p),
                    key="sidebar_provider_select"
                )
                if st.button("Apply & Switch", use_container_width=True):
                    try:
                        st.session_state.rag_system = initialize_rag_system(provider=selected_p)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            st.error("⚠️ System not initialized")
            if 'init_error' in st.session_state:
                st.error(st.session_state.init_error)
            if st.button("🔄 Initialize System", type="primary", use_container_width=True):
                try:
                    st.session_state.rag_system = initialize_rag_system()
                    st.session_state.pop('init_error', None)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📄 Documents", "💬 Chat", "ℹ️ About"])
    
    with tab1:
        st.header("Document Management")
        
        # Document upload section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                type=['pdf', 'txt', 'md'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("📤 Upload and Process Files"):
                    if not st.session_state.rag_system:
                        st.warning("Please initialize the RAG system first!")
                    else:
                        with st.spinner("Processing documents..."):
                            try:
                                # Save uploaded files temporarily
                                temp_dir = Path("temp_uploads")
                                temp_dir.mkdir(exist_ok=True)
                                
                                file_paths = []
                                for uploaded_file in uploaded_files:
                                    file_path = temp_dir / uploaded_file.name
                                    with open(file_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())
                                    file_paths.append(str(file_path))
                                
                                # Load documents
                                st.session_state.rag_system.load_from_files(file_paths)
                                st.session_state.documents_loaded = True
                                
                                # Update document list
                                st.session_state.document_list.extend([f.name for f in uploaded_files])
                                
                                st.success(f"Successfully loaded {len(uploaded_files)} document(s)!")
                                
                                # Clean up temp files
                                for file_path in file_paths:
                                    os.remove(file_path)
                                temp_dir.rmdir()
                                
                            except Exception as e:
                                st.error(f"Error loading documents: {str(e)}")
        
        with col2:
            st.subheader("Load from Directory")
            documents_dir = st.text_input("Documents Directory", "documents/")
            
            if st.button("📁 Load from Directory"):
                if not st.session_state.rag_system:
                    st.warning("Please initialize the RAG system first!")
                elif not os.path.exists(documents_dir):
                    st.error(f"Directory '{documents_dir}' does not exist!")
                else:
                    with st.spinner("Loading documents from directory..."):
                        try:
                            file_paths = load_documents_from_directory(documents_dir)
                            if file_paths:
                                st.session_state.rag_system.load_from_files(file_paths)
                                st.session_state.documents_loaded = True
                                st.session_state.document_list = [Path(p).name for p in file_paths]
                                st.success(f"Successfully loaded {len(file_paths)} document(s)!")
                            else:
                                st.warning("No supported documents found in the directory.")
                        except Exception as e:
                            st.error(f"Error loading documents: {str(e)}")
            
            st.divider()
            
            st.subheader("Load from ChromaDB")
            chroma_db_dir = st.text_input("ChromaDB Directory", "./chroma_db", key="chroma_db_dir")
            
            if st.button("💾 Load from ChromaDB", use_container_width=True):
                if not st.session_state.rag_system:
                    st.warning("Please initialize the RAG system first!")
                elif not os.path.exists(chroma_db_dir):
                    st.error(f"ChromaDB directory '{chroma_db_dir}' does not exist!")
                else:
                    with st.spinner("Loading from ChromaDB..."):
                        try:
                            success = st.session_state.rag_system.load_from_chromadb(chroma_db_dir)
                            if success:
                                st.session_state.documents_loaded = True
                                # Try to get document count
                                try:
                                    if st.session_state.rag_system.vectorstore:
                                        all_docs = st.session_state.rag_system.vectorstore.get()
                                        doc_count = len(all_docs.get('documents', [])) if all_docs else 0
                                        st.success(f"Successfully loaded ChromaDB with {doc_count} document chunks!")
                                    else:
                                        st.success("Successfully loaded ChromaDB!")
                                except:
                                    st.success("Successfully loaded ChromaDB!")
                            else:
                                st.error("Failed to load from ChromaDB. Check if the database exists and is valid.")
                        except Exception as e:
                            st.error(f"Error loading from ChromaDB: {str(e)}")
        
        # Display loaded documents
        if st.session_state.document_list:
            st.divider()
            st.subheader("📚 Loaded Documents")
            for i, doc in enumerate(st.session_state.document_list, 1):
                st.text(f"{i}. {doc}")
    
    with tab2:
        st.header("Chat with Your Documents")
        
        if not st.session_state.rag_system:
            st.info("👈 Please initialize the RAG system in the sidebar first!")
        elif not st.session_state.documents_loaded:
            st.info("📄 Please load documents in the 'Documents' tab first!")
        else:
            # Chat interface
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Show additional info if available
                    if message.get("metadata"):
                        with st.expander("📊 Query Details"):
                            metadata = message["metadata"]
                            
                            if metadata.get("routing"):
                                routing = metadata["routing"]
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Query Type", routing.get("query_type", "N/A"))
                                with col2:
                                    st.metric("Complexity", routing.get("complexity", "N/A"))
                                with col3:
                                    st.metric("Retrieval K", routing.get("recommended_k", "N/A"))
                            
                            if metadata.get("rewritten_query"):
                                st.caption(f"🔀 Rewritten Query: {metadata['rewritten_query']}")
                            
                            if metadata.get("validation"):
                                display_validation_results(metadata["validation"])
                            
                            if metadata.get("citations"):
                                st.caption("📎 Citations:")
                                st.text(format_citations(metadata["citations"]))
                            
                            if metadata.get("tools_used"):
                                st.caption(f"🛠️ Tools Used: {', '.join(metadata['tools_used'])}")
            
            # User input
            user_query = st.chat_input("Ask a question about your documents...")
            
            if user_query:
                # Add user message to chat
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_query
                })
                
                with st.chat_message("user"):
                    st.markdown(user_query)
                
                # Get response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            start_time = time.time()
                            result = st.session_state.rag_system.query(user_query)
                            elapsed_time = time.time() - start_time
                            
                            # Display answer
                            st.markdown(result['result'])
                            
                            # Collect metadata
                            metadata = {}
                            
                            if 'routing' in result:
                                metadata['routing'] = result['routing']
                            
                            if 'rewritten_query' in result and result['rewritten_query'] != user_query:
                                metadata['rewritten_query'] = result['rewritten_query']
                            
                            if 'validation' in result:
                                metadata['validation'] = result['validation']
                            
                            if 'citations' in result:
                                metadata['citations'] = result['citations']
                            
                            if 'tools_used' in result:
                                metadata['tools_used'] = result['tools_used']
                            
                            # Show metadata in expander
                            with st.expander("📊 Query Details"):
                                if metadata.get("routing"):
                                    routing = metadata["routing"]
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Query Type", routing.get("query_type", "N/A"))
                                    with col2:
                                        st.metric("Complexity", routing.get("complexity", "N/A"))
                                    with col3:
                                        st.metric("Retrieval K", routing.get("recommended_k", "N/A"))
                                
                                if metadata.get("rewritten_query"):
                                    st.caption(f"🔀 Rewritten Query: {metadata['rewritten_query']}")
                                
                                if metadata.get("validation"):
                                    display_validation_results(metadata["validation"])
                                
                                if metadata.get("is_general_answer"):
                                    st.info("🌐 **General Answer**: Not found in uploaded documents. Answer retrieved via Web Search / General Knowledge.")
                                
                                if metadata.get("citations"):
                                    st.caption("📎 Citations / Sources:")
                                    st.text(format_citations(metadata["citations"]))
                                
                                if metadata.get("tools_used"):
                                    st.caption(f"🛠️ Tools Used: {', '.join(metadata['tools_used'])}")
                                
                                st.caption(f"⏱️ Response Time: {elapsed_time:.2f}s")
                            
                            # Add assistant response to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": result['result'],
                                "metadata": metadata
                            })
                            
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
    
    with tab3:
        st.header("About the RAG System")
        
        st.markdown("""
        ### 🎯 Features
        
        This RAG (Retrieval-Augmented Generation) system includes:
        
        - **📚 Document Loading**: Support for PDF, TXT, and MD files
        - **🔍 Hybrid Search**: Combines vector similarity and BM25 keyword search
        - **🎯 Re-ranking**: Uses cross-encoder models to improve document relevance
        - **💭 Conversational Memory**: Maintains context across multiple queries
        - **🔄 Query Rewriting**: Automatically improves queries for better retrieval
        - **✅ Answer Validation**: Validates answers against source documents
        - **🧭 Query Routing**: Adapts retrieval strategy based on query type
        - **🛠️ External Tools**: Calculator and web search integration
        - **📎 Enhanced Citations**: Detailed source attribution with page numbers
        
        ### 🏗️ Architecture
        
        The system uses LangChain 1.2.3 with LCEL (LangChain Expression Language) for
        composable chains. It features a modular architecture with separate components
        for retrieval, memory, validation, and tool management.
        
        ### 📖 Usage
        
        1. Initialize the RAG system in the sidebar
        2. Upload or load documents in the Documents tab
        3. Start chatting in the Chat tab
        
        ### 🔧 Configuration
        
        All features can be toggled and configured in the sidebar. The system supports
        Google Gemini, Groq, and OpenAI providers.
        """)


if __name__ == "__main__":
    main()

