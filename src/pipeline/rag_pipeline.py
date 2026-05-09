"""
Main RAG Pipeline orchestrator
Brings together all components for a complete RAG system
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from ..config import RAGConfig, load_config
from ..models.llm_factory import LLMFactory
from ..data.document_loader import DocumentLoader
from ..data.text_splitter import DocumentSplitter
from ..vector_db.pinecone_client import PineconeClient
from ..retrieval.retrievers import create_pinecone_retriever
from ..chains.rag_chain import create_rag_chain, ConversationalRAG
from ..utils.logger import setup_logger


class RAGPipeline:
    """Complete RAG pipeline orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RAG pipeline
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logger("RAGPipeline", self.config)
        
        # Load environment variables
        self._load_environment()
        
        # Initialize components
        self.llm_factory = LLMFactory(self.config)
        self.document_loader = DocumentLoader(self.config)
        self.document_splitter = DocumentSplitter(self.config)
        
        # These will be initialized when needed
        self._pinecone_client = None
        self._retriever = None
        self._rag_chain = None
        self._conv_rag = None
        
        self.logger.info("RAG Pipeline initialized successfully")
    
    def _load_environment(self):
        """Load environment variables from configured file"""
        env_file = self.config.get_env_file()
        
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
            self.logger.info(f"Loaded environment from: {env_file}")
        else:
            self.logger.warning(f"Environment file not found: {env_file}")
        
        # Validate required API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not self.openai_api_key or not self.pinecone_api_key:
            missing = []
            if not self.openai_api_key:
                missing.append("OPENAI_API_KEY")
            if not self.pinecone_api_key:
                missing.append("PINECONE_API_KEY")
            
            raise EnvironmentError(
                f"Missing required environment variables: {missing}. "
                f"Add them to {self.config.get_env_file()} or your system environment."
            )
        
        # Set environment variables for downstream libraries
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
    
    def get_pinecone_client(self) -> PineconeClient:
        """Get or create Pinecone client"""
        if self._pinecone_client is None:
            self._pinecone_client = PineconeClient(self.config, self.pinecone_api_key)
            self.logger.info("Pinecone client initialized")
        return self._pinecone_client
    
    def setup_vector_db(self):
        """Setup vector database (create index if needed)"""
        pinecone_client = self.get_pinecone_client()
        pinecone_client.create_index_if_needed()
        self.logger.info("Vector database setup completed")
    
    def load_and_process_documents(self) -> List[Any]:
        """Load and process documents into chunks"""
        self.logger.info("Loading documents...")
        documents = self.document_loader.load_pdf_files()
        
        self.logger.info("Splitting documents into chunks...")
        chunks = self.document_splitter.split_documents(documents)
        
        # Log statistics
        stats = self.document_splitter.get_chunk_stats(chunks)
        self.logger.info(f"Document processing completed: {stats}")
        
        return chunks
    
    def ingest_documents(self, chunks: List[Any]):
        """Ingest document chunks into vector database"""
        self.logger.info("Starting document ingestion...")
        
        pinecone_client = self.get_pinecone_client()
        embeddings = self.llm_factory.create_embedding_model()
        
        pinecone_client.upsert_documents(chunks, embeddings)
        self.logger.info("Document ingestion completed")
    
    def get_retriever(self):
        """Get or create document retriever"""
        if self._retriever is None:
            pinecone_client = self.get_pinecone_client()
            embeddings = self.llm_factory.create_embedding_model()
            
            self._retriever = create_pinecone_retriever(
                config=self.config,
                pinecone_client=pinecone_client,
                embeddings=embeddings
            )
            self.logger.info("Retriever created")
        
        return self._retriever
    
    def get_rag_chain(self):
        """Get or create RAG chain"""
        if self._rag_chain is None:
            retriever = self.get_retriever()
            llm = self.llm_factory.create_chat_model()
            
            self._rag_chain = create_rag_chain(
                config=self.config,
                retriever=retriever,
                llm=llm
            )
            self.logger.info("RAG chain created")
        
        return self._rag_chain
    
    def get_conversational_rag(self) -> ConversationalRAG:
        """Get or create conversational RAG system"""
        if self._conv_rag is None:
            rag_chain = self.get_rag_chain()
            self._conv_rag = ConversationalRAG(rag_chain, self.config)
            self.logger.info("Conversational RAG created")
        
        return self._conv_rag
    
    def ask_question(self, question: str, show_sources: bool = None) -> Dict[str, Any]:
        """
        Ask a question using the RAG system
        
        Args:
            question: The question to ask
            show_sources: Whether to show sources in output
            
        Returns:
            Dictionary with answer and context
        """
        if show_sources is None:
            show_sources = self.config.conversation.show_sources
        
        rag_chain = self.get_rag_chain()
        response = rag_chain.invoke({"input": question})
        
        if show_sources:
            self._print_response_with_sources(question, response)
        else:
            print(f"Question: {question}")
            print(f"Answer: {response['answer']}")
        
        return response
    
    def _print_response_with_sources(self, question: str, response: Dict[str, Any]):
        """Print response with formatted sources"""
        print(f"Question: {question}")
        print(f"\nAnswer: {response['answer']}")
        
        print("\nRetrieved sources:")
        for i, doc in enumerate(response.get("context", []), start=1):
            source_file = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            score = doc.metadata.get('score', 0)
            print(f"  {i}. {source_file} (Page {page}) - Score: {score:.4f}")
    
    def full_setup(self):
        """Run complete pipeline setup"""
        self.logger.info("Starting full pipeline setup...")
        
        # Setup vector database
        self.setup_vector_db()
        
        # Load and process documents
        chunks = self.load_and_process_documents()
        
        # Ingest documents
        self.ingest_documents(chunks)
        
        # Initialize RAG chain
        self.get_rag_chain()
        
        self.logger.info("Full pipeline setup completed successfully!")
    
    def test_retrieval(self, query: str = "What is investment?", k: int = 3):
        """Test document retrieval"""
        retriever = self.get_retriever()
        docs = retriever.invoke(query)
        
        print(f"Test Query: {query}")
        print(f"Retrieved {len(docs)} documents:")
        
        for i, doc in enumerate(docs[:k], 1):
            source_file = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            score = doc.metadata.get('score', 0)
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            print(f"\n--- Document {i} ---")
            print(f"Source: {source_file} (Page {page})")
            print(f"Score: {score:.4f}")
            print(f"Content: {content_preview}")


# Factory function for easy pipeline creation
def create_pipeline(config_path: Optional[str] = None) -> RAGPipeline:
    """Create a RAG pipeline instance"""
    return RAGPipeline(config_path)