"""
Enhanced Streamlit RAG Application with Configuration Management
Uses modular architecture with YAML configuration
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add src directory to Python path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR / "src"))

try:
    # New modular imports
    from src.config import load_config
    from src.pipeline.rag_pipeline import create_pipeline
    from src.utils.logger import setup_logger
    MODULAR_AVAILABLE = True
except ImportError:
    # Fallback to legacy approach
    st.warning("⚠️ Modular components not available. Using legacy approach.")
    MODULAR_AVAILABLE = False
    
    # Legacy imports (simplified for this demo)
    import os
    from dotenv import load_dotenv
    from pinecone import Pinecone
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system - uses modular approach if available, legacy otherwise"""
    
    if MODULAR_AVAILABLE:
        return _initialize_modular_system()
    else:
        return _initialize_legacy_system()


def _initialize_modular_system():
    """Initialize using the new modular architecture"""
    try:
        # Load configuration
        config = load_config()
        
        # Create pipeline
        pipeline = create_pipeline()
        
        # Setup vector database (cached)
        if not hasattr(st.session_state, 'vector_db_ready'):
            with st.spinner("Setting up vector database..."):
                pipeline.setup_vector_db()
            st.session_state.vector_db_ready = True
        
        # Get conversational RAG
        conv_rag = pipeline.get_conversational_rag()
        
        return {
            'conv_rag': conv_rag,
            'pipeline': pipeline,
            'config': config,
            'type': 'modular'
        }
        
    except Exception as e:
        st.error(f"Failed to initialize modular system: {e}")
        raise


def _initialize_legacy_system():
    """Fallback to legacy initialization"""
    st.info("📚 Using legacy system (modular components not available)")
    
    # This would contain the legacy initialization code
    # For now, we'll show an error since the user should use the modular approach
    st.error("Legacy system not implemented. Please set up the modular architecture.")
    st.stop()


def render_sidebar(config=None):
    """Render the sidebar with controls and information"""
    with st.sidebar:
        st.header("🔧 Configuration")
        
        if config and hasattr(config, 'models'):
            st.markdown(f"**Embedding Model:** {config.models.embedding_model}")
            st.markdown(f"**Chat Model:** {config.models.chat_model}")
            st.markdown(f"**Temperature:** {config.models.temperature}")
            st.markdown(f"**Retrieval K:** {config.retrieval.k}")
        
        st.divider()
        
        st.header("💬 Chat Controls")
        
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("🔄 Restart System", type="secondary"):
            st.cache_resource.clear()
            st.rerun()
        
        st.divider()
        
        st.header("ℹ️ Instructions")
        st.markdown("""
        - Ask any question about personal finance
        - I can handle follow-up questions using context
        - I cite sources for all answers
        - Use **Clear Conversation** to start fresh
        - Use **Restart System** if you encounter issues
        """)
        
        if st.session_state.get('chat_history', []):
            st.divider()
            st.header("📊 Stats")
            st.metric("Messages", len(st.session_state.chat_history))
        
        st.divider()
        st.header("🚀 Quick Examples")
        example_questions = [
            "What is Warren Buffett's investment philosophy?",
            "How should I invest $1000?",
            "What is a margin of safety?",
            "Explain diversification strategies",
            "What are financial ratios?"
        ]
        
        for question in example_questions:
            if st.button(f"💡 {question[:30]}...", key=f"example_{hash(question)}"):
                st.session_state.example_question = question


def render_chat_message(role: str, content: str, sources: Optional[List] = None):
    """Render a chat message with sources"""
    with st.chat_message(role):
        st.write(content)
        
        if sources and role == "assistant":
            with st.expander("📚 Sources", expanded=False):
                for i, doc in enumerate(sources, 1):
                    source_file = doc.metadata.get('source_file', 'Unknown')
                    page = doc.metadata.get('page', 'Unknown')
                    score = doc.metadata.get('score', 0)
                    
                    st.markdown(f"""
                    **Source {i}:** {source_file} (Page {page})  
                    **Relevance Score:** {score:.4f}  
                    **Preview:** {doc.page_content[:200]}...
                    """)


def process_question(question: str, conv_rag, chat_history: List[Tuple[str, str]]) -> Tuple[str, List]:
    """Process a question and return answer with sources"""
    with st.spinner("🤔 Thinking..."):
        try:
            response = conv_rag.ask(question, chat_history)
            answer = response["answer"]
            sources = response.get("context", [])
            
            return answer, sources
            
        except Exception as e:
            st.error(f"Error processing question: {e}")
            return f"I'm sorry, I encountered an error: {e}", []


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Personal Finance RAG Assistant",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "example_question" not in st.session_state:
        st.session_state.example_question = None
    
    # Initialize RAG system
    try:
        rag_system = initialize_rag_system()
        conv_rag = rag_system['conv_rag']
        config = rag_system.get('config')
        
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.stop()
    
    # Render sidebar
    render_sidebar(config)
    
    # Main content
    st.title("💰 Personal Finance RAG Assistant")
    
    if MODULAR_AVAILABLE:
        st.success("✅ Using enhanced modular architecture with YAML configuration")
    else:
        st.info("📚 Using legacy architecture")
    
    st.markdown("""
    Ask me questions about personal finance! I can provide answers based on your documents 
    and handle follow-up questions that reference previous conversations.
    """)
    
    # Display chat history
    for question, answer in st.session_state.chat_history:
        render_chat_message("user", question)
        # Note: We don't have sources stored in history, could enhance this
        render_chat_message("assistant", answer)
    
    # Handle example question from sidebar
    question_to_process = None
    if st.session_state.example_question:
        question_to_process = st.session_state.example_question
        st.session_state.example_question = None  # Clear after use
        
        # Show the example question as if user typed it
        render_chat_message("user", question_to_process)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about personal finance..."):
        question_to_process = prompt
        render_chat_message("user", prompt)
    
    # Process question if we have one
    if question_to_process:
        answer, sources = process_question(question_to_process, conv_rag, st.session_state.chat_history)
        
        # Display assistant response
        render_chat_message("assistant", answer, sources)
        
        # Add to chat history
        st.session_state.chat_history.append((question_to_process, answer))
        
        # Auto-scroll to bottom (rerun to show new message)
        st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: small;'>
    💡 This assistant uses RAG (Retrieval-Augmented Generation) to provide answers based on your documents.<br>
    🔧 Enhanced with modular architecture and YAML configuration management.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()