"""
Text splitting functionality for document chunking
Creates smaller, manageable chunks for better retrieval
"""

import hashlib
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..config import RAGConfig


class DocumentSplitter:
    """Handles splitting documents into smaller chunks"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.document_processing.chunk_size,
            chunk_overlap=config.document_processing.chunk_overlap,
            separators=config.document_processing.separators,
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks with unique IDs and metadata
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # Add stable chunk IDs and enhanced metadata
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown_source")
            page = chunk.metadata.get("page", "unknown_page")
            content_hash = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()[:12]

            # Create unique chunk ID
            chunk.metadata["chunk_id"] = f"{Path(str(source)).stem}-p{page}-c{i}-{content_hash}"
            chunk.metadata["source_file"] = source
            chunk.metadata["chunk_index"] = i

        print(f"Created {len(chunks)} text chunk(s).")
        return chunks
    
    def preview_chunk(self, chunks: List[Document], index: int = 0) -> None:
        """
        Preview a chunk's metadata and content
        
        Args:
            chunks: List of chunks
            index: Index of chunk to preview
        """
        if not chunks or index >= len(chunks):
            print("No chunks to preview or invalid index")
            return
        
        chunk = chunks[index]
        print(f"\nChunk Preview (#{index + 1}):")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content preview: {chunk.page_content[:500]}...")
    
    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about the chunks
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {"total_chunks": 0}
        
        lengths = [len(chunk.page_content) for chunk in chunks]
        sources = set(chunk.metadata.get("source_file", "unknown") for chunk in chunks)
        
        return {
            "total_chunks": len(chunks),
            "average_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "unique_sources": len(sources),
            "source_files": list(sources)
        }