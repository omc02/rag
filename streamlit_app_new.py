"""
Enhanced Streamlit RAG Application 
Supports both modular (recommended) and legacy approaches
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src directory to Python path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR / "src"))

# Always import basic dependencies
import os
import time
import hashlib
from dotenv import load_dotenv

try:
    # New modular imports
    from src.config import load_config
    from src.pipeline.rag_pipeline import create_pipeline
    MODULAR_AVAILABLE = True
except ImportError:
    # Fallback to legacy imports when modular approach fails
    try:
        from pinecone import Pinecone, ServerlessSpec
        from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from pydantic import PrivateAttr
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
        LEGACY_IMPORTS_AVAILABLE = True
    except ImportError:
        LEGACY_IMPORTS_AVAILABLE = False
    MODULAR_AVAILABLE = False

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system - uses modular approach if available"""
    
    if MODULAR_AVAILABLE:
        try:
            # Load configuration
            config = load_config()
            
            # Create pipeline
            pipeline = create_pipeline()
            
            # Setup vector database (cached)
            with st.spinner("Setting up RAG system..."):
                pipeline.setup_vector_db()
            
            # Get conversational RAG
            conv_rag = pipeline.get_conversational_rag()
            
            return conv_rag, config, "modular"
            
        except Exception as e:
            st.error(f"Failed to initialize modular system: {e}")
            st.info("Falling back to legacy approach...")
    
    # Legacy fallback
    return initialize_legacy_system()

def initialize_legacy_system():
    """Legacy initialization for backward compatibility"""
    
    if not globals().get('LEGACY_IMPORTS_AVAILABLE', False):
        st.error("❌ Neither modular nor legacy dependencies are available.")
        st.error("Please install requirements: pip install -r requirements_rag_fixed.txt")
        st.stop()
    
    # Load environment variables
    load_dotenv("env/api_keys.env")
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        st.error("Please ensure OPENAI_API_KEY and PINECONE_API_KEY are set in env/api_keys.env")
        st.stop()
    
    # Simple legacy implementation for demo
    class SimpleLegacyRAG:
        def ask(self, question: str, chat_history: list) -> Dict[str, Any]:
            # Basic response for legacy mode
            return {
                "answer": "⚠️ Legacy mode active. For full functionality, please:\n"
                         "1. Install all requirements: pip install -r requirements_rag_fixed.txt\n"
                         "2. Ensure your modular architecture is set up properly\n"
                         "3. Run: python test_config.py to verify setup\n\n"
                         f"Your question was: {question}",
                "context": []
            }
    
    # Return simplified objects for legacy mode
    conv_rag = SimpleLegacyRAG()
    config = None
    
    return conv_rag, config, "legacy"

def main():
    st.set_page_config(
        page_title="Personal Finance RAG Assistant",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Personal Finance RAG Assistant")
    
    # Initialize RAG system
    try:
        conv_rag, config, system_type = initialize_rag_system()
        
        if system_type == "modular":
            st.success("✅ Using enhanced modular architecture with YAML configuration")
        else:
            st.info("📚 Using legacy architecture")
            
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.stop()
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "example_question" not in st.session_state:
        st.session_state.example_question = None
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        if system_type == "modular":
            st.success("✅ Modular Architecture")
            if config and hasattr(config, 'models'):
                st.markdown(f"**Embedding:** {config.models.embedding_model}")
                st.markdown(f"**Chat:** {config.models.chat_model}")
                st.markdown(f"**Temperature:** {config.models.temperature}")
                st.markdown(f"**Retrieval K:** {config.retrieval.k}")
        else:
            st.warning("⚠️ Legacy Mode")
            st.markdown("For full features, run:")
            st.code("python test_config.py")
        
        st.divider()
        
        st.header("💬 Controls")
        
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("🔄 Restart System", type="secondary"):
            st.cache_resource.clear()
            st.rerun()
        
        st.divider()
        
        st.header("💡 Examples")
        examples = [
            "What is Warren Buffett's investment philosophy?",
            "How should I invest $1000?",
            "What is a margin of safety?",
            "Explain diversification strategies"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"📝 {example[:25]}...", key=f"ex_{i}"):
                st.session_state.example_question = example
        
        if st.session_state.chat_history:
            st.divider()
            st.metric("Messages", len(st.session_state.chat_history))
    
    # Main content
    st.markdown("Ask questions about personal finance and get answers from your documents!")
    
    # Display chat history
    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
    
    # Handle example questions
    question_to_process = None
    if st.session_state.example_question:
        question_to_process = st.session_state.example_question
        st.session_state.example_question = None
        with st.chat_message("user"):
            st.write(question_to_process)
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        question_to_process = prompt
        with st.chat_message("user"):
            st.write(prompt)
    
    # Process question
    if question_to_process:
        with st.spinner("Thinking..."):
            try:
                response = conv_rag.ask(question_to_process, st.session_state.chat_history)
                answer = response["answer"]
                sources = response.get("context", [])
                
                with st.chat_message("assistant"):
                    st.write(answer)
                    
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(sources[:3], 1):
                                source = doc.metadata.get('source_file', 'Unknown')
                                page = doc.metadata.get('page', 'Unknown')
                                score = doc.metadata.get('score', 0)
                                st.markdown(f"**{i}.** {source} (Page {page}) - Score: {score:.3f}")
                
                # Add to history
                st.session_state.chat_history.append((question_to_process, answer))
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()