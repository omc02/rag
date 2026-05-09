"""
Custom retrievers for the RAG pipeline
Implements Pinecone-based retrieval without requiring langchain-pinecone
"""

from typing import Any, Dict, List, Optional
from pydantic import PrivateAttr
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_openai import OpenAIEmbeddings

from ..config import RAGConfig
from ..vector_db.pinecone_client import PineconeClient


class PineconeRetriever(BaseRetriever):
    """Custom Pinecone retriever that integrates with LangChain"""
    
    k: int = 5
    namespace: Optional[str] = None
    search_filter: Optional[Dict[str, Any]] = None

    _pinecone_client: Any = PrivateAttr()
    _embeddings: Any = PrivateAttr()
    _config: Any = PrivateAttr()

    def __init__(
        self,
        pinecone_client: PineconeClient,
        embeddings: OpenAIEmbeddings,
        config: RAGConfig,
        k: int = None,
        namespace: Optional[str] = None,
        search_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if k is None:
            k = config.retrieval.k
            
        if namespace is None:
            namespace = config.vector_db.namespace
            
        super().__init__(
            k=k,
            namespace=namespace,
            search_filter=search_filter,
            **kwargs,
        )
        self._pinecone_client = pinecone_client
        self._embeddings = embeddings
        self._config = config

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant documents for a query"""
        
        # Generate query embedding
        query_vector = self._embeddings.embed_query(query)
        
        # Query Pinecone
        matches = self._pinecone_client.query_similar(
            query_vector=query_vector,
            k=self.k,
            namespace=self.namespace,
            filter_dict=self.search_filter
        )
        
        # Convert matches to Documents
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


class HybridRetriever(BaseRetriever):
    """Hybrid retriever that combines multiple retrieval methods"""
    
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
        k: int = 5,
        **kwargs: Any,
    ):
        super().__init__(k=k, **kwargs)
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        
        if len(self.weights) != len(self.retrievers):
            raise ValueError("Number of weights must match number of retrievers")
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Combine results from multiple retrievers"""
        
        all_docs = []
        doc_scores = {}
        
        # Get results from each retriever
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.get_relevant_documents(query, callbacks=run_manager.get_child())
            
            for doc in docs:
                doc_id = doc.metadata.get("chunk_id", id(doc))
                base_score = doc.metadata.get("score", 0.5)
                weighted_score = base_score * weight
                
                if doc_id in doc_scores:
                    # Combine scores (you could use different combination strategies)
                    doc_scores[doc_id] = max(doc_scores[doc_id], weighted_score)
                else:
                    doc_scores[doc_id] = weighted_score
                    all_docs.append(doc)
        
        # Sort by combined score and return top k
        scored_docs = []
        for doc in all_docs:
            doc_id = doc.metadata.get("chunk_id", id(doc))
            doc.metadata["combined_score"] = doc_scores[doc_id]
            scored_docs.append(doc)
        
        scored_docs.sort(key=lambda x: x.metadata.get("combined_score", 0), reverse=True)
        return scored_docs[:self.k]


def create_pinecone_retriever(
    config: RAGConfig,
    pinecone_client: PineconeClient,
    embeddings: OpenAIEmbeddings,
    **kwargs
) -> PineconeRetriever:
    """
    Factory function to create a configured Pinecone retriever
    
    Args:
        config: RAG configuration
        pinecone_client: Pinecone client instance
        embeddings: Embedding model
        **kwargs: Additional arguments for retriever
        
    Returns:
        Configured PineconeRetriever instance
    """
    return PineconeRetriever(
        pinecone_client=pinecone_client,
        embeddings=embeddings,
        config=config,
        **kwargs
    )