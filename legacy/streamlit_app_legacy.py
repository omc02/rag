"""
Enhanced Streamlit RAG Application 
Supports both modular (recommended) and legacy approaches
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Always import these for compatibility
import os
import time
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import PrivateAttr
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Add src directory to Python path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR / "src"))

try:
    # New modular imports
    from src.config import load_config
    from src.pipeline.rag_pipeline import create_pipeline
    MODULAR_AVAILABLE = True
    st.sidebar.success("✅ Using modular architecture")
except ImportError:
    MODULAR_AVAILABLE = False
    st.sidebar.info("📚 Using legacy architecture")

# Configuration and Pipeline Setup
@st.cache_resource
def get_rag_pipeline():
    """Initialize and cache the RAG pipeline"""
    if MODULAR_AVAILABLE:
        try:
            # Use new modular approach
            config = load_config()
            pipeline = create_pipeline()
            
            # Check if we need to run full setup
            if not hasattr(st.session_state, 'pipeline_ready'):
                with st.spinner("Setting up RAG pipeline..."):
                    pipeline.setup_vector_db()
                st.session_state.pipeline_ready = True
            
            return pipeline, config, True
        except Exception as e:
            st.error(f"Failed to initialize modular pipeline: {e}")
            st.info("Falling back to legacy approach...")
    
    # Fallback to legacy approach
    return None, None, False

# Legacy classes for backward compatibility (only used when modular approach fails)
if not MODULAR_AVAILABLE:
    class StuffDocumentsChain:
        """
        Custom implementation of create_stuff_documents_chain with invoke method
        """
    
    def __init__(self, llm, prompt, document_prompt=None):
        self.llm = llm
        self.prompt = prompt
        self.document_prompt = document_prompt
    
    def format_docs(self, docs):
        if self.document_prompt:
            formatted = []
            for doc in docs:
                formatted.append(self.document_prompt.format(**doc.metadata, page_content=doc.page_content))
            return "\n\n".join(formatted)
        else:
            return "\n\n".join(doc.page_content for doc in docs)
    
    def invoke(self, inputs):
        docs = inputs["context"]
        context = self.format_docs(docs)
        # Create a new dict with the formatted context, avoiding duplicate keys
        prompt_inputs = {k: v for k, v in inputs.items() if k != "context"}
        prompt_inputs["context"] = context
        messages = self.prompt.format_messages(**prompt_inputs)
        response = self.llm.invoke(messages)
        return {"answer": response.content, "context": docs}

class RetrievalChain:
    """Custom implementation of create_retrieval_chain with invoke method"""
    
    def __init__(self, retriever, combine_docs_chain):
        self.retriever = retriever
        self.combine_docs_chain = combine_docs_chain
    
    def invoke(self, inputs):
        # Get the query
        query = inputs["input"]
        
        # Retrieve relevant documents
        docs = self.retriever.invoke(query)
        
        # Pass to the combine chain
        chain_input = {"context": docs, "input": query}
        result = self.combine_docs_chain.invoke(chain_input)
        
        return result

def create_stuff_documents_chain(llm, prompt, document_prompt=None):
    """Custom implementation of create_stuff_documents_chain"""
    return StuffDocumentsChain(llm, prompt, document_prompt)

def create_retrieval_chain(retriever, combine_docs_chain):
    """Custom implementation of create_retrieval_chain"""
    return RetrievalChain(retriever, combine_docs_chain)

class PineconeRetriever(BaseRetriever):
    k: int = 5
    namespace: Optional[str] = None
    search_filter: Optional[Dict[str, Any]] = None

    _index: Any = PrivateAttr()
    _embeddings: Any = PrivateAttr()

    def __init__(
        self,
        index: Any,
        embeddings: OpenAIEmbeddings,
        k: int = 5,
        namespace: Optional[str] = None,
        search_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            k=k,
            namespace=namespace,
            search_filter=search_filter,
            **kwargs,
        )
        self._index = index
        self._embeddings = embeddings

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        query_vector = self._embeddings.embed_query(query)

        query_kwargs = {
            "vector": query_vector,
            "top_k": self.k,
            "include_metadata": True,
        }

        if self.namespace:
            query_kwargs["namespace"] = self.namespace

        if self.search_filter:
            query_kwargs["filter"] = self.search_filter

        results = self._index.query(**query_kwargs)

        matches = getattr(results, "matches", None)
        if matches is None and isinstance(results, dict):
            matches = results.get("matches", [])

        retrieved_docs = []

        for match in matches:
            metadata = getattr(match, "metadata", None)
            score = getattr(match, "score", None)

            if metadata is None and isinstance(match, dict):
                metadata = match.get("metadata", {})
                score = match.get("score", score)

            metadata = dict(metadata or {})
            text = metadata.pop("text", "")

            metadata["score"] = float(score) if score is not None else None

            retrieved_docs.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        return retrieved_docs

class ConversationalRAG:
    """
    A conversational RAG system that maintains chat history and can handle follow-up questions.
    """
    
    def __init__(self, rag_chain, max_history=5):
        self.rag_chain = rag_chain
        self.max_history = max_history
    
    def _format_chat_history(self, chat_history):
        """Format recent chat history for context"""
        if not chat_history:
            return ""
        
        history_text = "\n\nRecent conversation history:\n"
        for i, (q, a) in enumerate(chat_history[-self.max_history:], 1):
            history_text += f"Q{i}: {q}\n"
            history_text += f"A{i}: {a[:200]}{'...' if len(a) > 200 else ''}\n\n"
        
        return history_text
    
    def ask(self, question: str, chat_history: list) -> Dict[str, Any]:
        """
        Ask a question with conversation history context
        """
        # Add conversation context to the question if there's history
        contextual_question = question
        if chat_history:
            contextual_question = f"""
Previous conversation context:
{self._format_chat_history(chat_history)}

Current question: {question}

Please answer the current question, taking into account the previous conversation context if relevant. If the question refers to something mentioned earlier (like "it", "that", "the company", etc.), use the conversation history to understand what the user is referring to.
"""
        
        # Get response from RAG chain
        response = self.rag_chain.invoke({"input": contextual_question})
        return response

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached for performance)"""
    
    # Load environment variables
    load_dotenv("env/api_keys.env")
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        st.error("Please ensure OPENAI_API_KEY and PINECONE_API_KEY are set in env/api_keys.env")
        st.stop()
    
    # Settings
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o-mini"
    INDEX_NAME = "rag-3-pdf-openai"
    NAMESPACE = "personal-finance-pdfs"
    
    # Initialize OpenAI and Pinecone
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    # Create retriever (increased k for better coverage)
    retriever = PineconeRetriever(
        index=index,
        embeddings=embeddings,
        k=10,
        namespace=NAMESPACE,
    )
    
    # Create RAG chain (improved settings for better advice)
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.1,
        max_tokens=1200,
    )
    
    system_prompt = """
You are a helpful personal finance knowledge assistant.

Answer the user's question by synthesizing relevant information from the retrieved context below. When users ask for investment advice or recommendations, apply the principles, strategies, and wisdom found in the documents to provide practical, actionable guidance.

Rules:
1. Base your answers on the principles and strategies described in the retrieved context
2. For investment/financial advice questions, synthesize relevant principles into practical recommendations
3. Always cite specific source files and page numbers for the principles you reference
4. If you can apply document principles to answer the question, do so - don't just say "I don't know"
5. Provide concrete, actionable advice when the documents contain relevant investment principles
6. If the documents truly contain no relevant information, then say you cannot find relevant guidance
7. Structure advice clearly: start with key principles, then specific recommendations, then cite sources

Retrieved context:
{context}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    document_prompt = PromptTemplate.from_template(
        "Source file: {source_file}\n"
        "Page: {page}\n"
        "Chunk ID: {chunk_id}\n"
        "Content:\n{page_content}"
    )
    
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=document_prompt,
    )
    
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    
    # Create conversational RAG
    conv_rag = ConversationalRAG(rag_chain, max_history=5)
    
    return conv_rag

def main():
    st.set_page_config(
        page_title="Personal Finance RAG Assistant",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Personal Finance RAG Assistant")
    st.markdown("Ask questions about personal finance and get answers from your documents. I can handle follow-up questions too!")
    
    # Initialize RAG system
    try:
        conv_rag = initialize_rag_system()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        st.stop()
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "example_question" not in st.session_state:
        st.session_state.example_question = None
    
    # Sidebar with controls
    with st.sidebar:
        st.header("Chat Controls")
        
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.divider()
        
        st.header("Instructions")
        st.markdown("""
        - Ask any question about personal finance
        - I can answer follow-up questions that refer to previous answers
        - Use 'Clear Conversation' to start fresh
        - I cite sources for all my answers
        """)
        
        if st.session_state.chat_history:
            st.divider()
            st.header("Conversation Stats")
            st.metric("Messages", len(st.session_state.chat_history))
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask a question:",
            placeholder="e.g., What is Warren Buffett's investment philosophy?",
            key="user_input"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    # Handle example question clicks
    question_to_process = None
    if hasattr(st.session_state, 'example_question') and st.session_state.example_question:
        question_to_process = st.session_state.example_question
        st.session_state.example_question = None  # Clear it after using
    elif (ask_button or user_question) and user_question.strip():
        question_to_process = user_question

    # Process question
    if question_to_process:
        with st.spinner("Getting answer..."):
            try:
                response = conv_rag.ask(question_to_process, st.session_state.chat_history)
                answer = response["answer"]
                sources = response.get("context", [])
                
                # Add to chat history
                st.session_state.chat_history.append((question_to_process, answer))
                
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")
                return
    
    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        st.header("Conversation")
        
        for i, (question, answer) in enumerate(reversed(st.session_state.chat_history)):
            message_num = len(st.session_state.chat_history) - i
            
            with st.container():
                # Question
                st.markdown(f"**🙋 Question {message_num}:** {question}")
                
                # Answer
                st.markdown(f"**🤖 Answer:** {answer}")
                
                # Sources (for the most recent question only)
                if i == 0 and 'response' in locals():
                    with st.expander("📚 Sources"):
                        sources = response.get("context", [])
                        for j, doc in enumerate(sources, 1):
                            source_file = doc.metadata.get('source_file', 'Unknown')
                            page = doc.metadata.get('page', 'Unknown')
                            score = doc.metadata.get('score', 0)
                            
                            st.markdown(f"**{j}.** {source_file} (Page {page}) - Relevance: {score:.3f}")
                
                st.divider()
    else:
        # Welcome message
        st.info("👋 Welcome! Ask me any question about personal finance. I can handle follow-up questions and maintain context throughout our conversation.")
        
        # Example questions
        st.subheader("💡 Example Questions:")
        example_questions = [
            "What is Warren Buffett's investment philosophy?",
            "What is a PE ratio?",
            "How do I value a stock?",
            "What is the margin of safety principle?",
            "Should I invest $1000 in stocks? Why?"
        ]
        
        for eq in example_questions:
            if st.button(f"💬 {eq}", key=f"example_{hash(eq)}"):
                st.session_state.example_question = eq
                st.rerun()

if __name__ == "__main__":
    main()