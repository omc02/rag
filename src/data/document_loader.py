"""
Document loading functionality for the RAG pipeline
Handles PDF loading and basic preprocessing
"""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document

from ..config import RAGConfig


class DocumentLoader:
    """Handles loading documents from various sources"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def list_pdf_files(self, pdf_dir: Path = None) -> List[Path]:
        """
        List all PDF files in the configured directory
        
        Args:
            pdf_dir: Override default PDF directory
            
        Returns:
            List of PDF file paths
        """
        if pdf_dir is None:
            pdf_dir = self.config.get_pdf_dir()
        
        if not pdf_dir.exists():
            raise FileNotFoundError(
                f"PDF directory does not exist: {pdf_dir}\n"
                f"Create this folder and place your PDF files inside it."
            )

        pdf_files = sorted(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

        print(f"Found {len(pdf_files)} PDF file(s):")
        for pdf in pdf_files:
            print(f" - {pdf.name}")

        return pdf_files
    
    def load_pdf_files(self, pdf_dir: Path = None) -> List[Document]:
        """
        Load all PDF files from directory
        
        Args:
            pdf_dir: Override default PDF directory
            
        Returns:
            List of Document objects (one per page)
        """
        if pdf_dir is None:
            pdf_dir = self.config.get_pdf_dir()
        
        loader = DirectoryLoader(
            str(pdf_dir),
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )

        documents = loader.load()

        # Normalize source path to file name for cleaner citations
        for doc in documents:
            source = doc.metadata.get("source", "")
            doc.metadata["source"] = Path(source).name if source else "unknown_source"
            doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1  # convert to 1-based page numbers

        print(f"Loaded {len(documents)} page-level document(s).")
        return documents
    
    def preview_document(self, documents: List[Document], index: int = 0) -> None:
        """
        Preview a document's metadata and content
        
        Args:
            documents: List of documents
            index: Index of document to preview
        """
        if not documents or index >= len(documents):
            print("No documents to preview or invalid index")
            return
        
        doc = documents[index]
        print(f"\nDocument Preview (#{index + 1}):")
        print(f"Metadata: {doc.metadata}")
        print(f"Content preview: {doc.page_content[:500]}...")