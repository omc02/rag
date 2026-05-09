"""
Pinecone vector database client and operations
Handles index creation, upserting, and querying
"""

import time
from typing import Any, Dict, List, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ..config import RAGConfig


class PineconeClient:
    """Client for interacting with Pinecone vector database"""
    
    def __init__(self, config: RAGConfig, api_key: str):
        self.config = config
        self.pc = Pinecone(api_key=api_key)
        self._index = None
    
    def get_index_names(self) -> List[str]:
        """Get list of existing Pinecone indexes"""
        indexes = self.pc.list_indexes()

        # Newer SDKs often expose .names()
        if hasattr(indexes, "names"):
            return indexes.names()

        # Fallback for iterable index objects / dictionaries
        names = []
        for idx in indexes:
            if isinstance(idx, dict):
                names.append(idx.get("name"))
            else:
                names.append(getattr(idx, "name", None))

        return [name for name in names if name]
    
    def create_index_if_needed(self) -> None:
        """Create Pinecone index if it doesn't exist"""
        existing_index_names = self.get_index_names()
        index_name = self.config.vector_db.index_name
        dimension = self.config.models.embedding_dimension

        if index_name in existing_index_names:
            print(f"Using existing Pinecone index: {index_name}")

            # Validate dimension if possible
            try:
                description = self.pc.describe_index(index_name)
                existing_dimension = getattr(description, "dimension", None)

                if existing_dimension is None and isinstance(description, dict):
                    existing_dimension = description.get("dimension")

                if existing_dimension is not None and int(existing_dimension) != int(dimension):
                    raise ValueError(
                        f"Existing index '{index_name}' has dimension {existing_dimension}, "
                        f"but embedding model requires dimension {dimension}. "
                        "Use a new index name or delete/recreate the existing index."
                    )
            except AttributeError:
                # Older/newer SDK surface difference; skip dimension validation
                pass

            return

        print(f"Creating Pinecone index: {index_name}")

        spec = ServerlessSpec(
            cloud=self.config.vector_db.cloud,
            region=self.config.vector_db.region
        )

        # Create index
        if hasattr(self.pc, "create_index"):
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=self.config.vector_db.metric,
                spec=spec,
            )
        else:
            # Fallback for different SDK versions
            self.pc.indexes.create(
                name=index_name,
                dimension=dimension,
                metric=self.config.vector_db.metric,
                spec=spec,
            )

        # Wait until ready
        self._wait_for_index_ready(index_name)
    
    def _wait_for_index_ready(self, index_name: str, max_wait: int = 120) -> None:
        """Wait for index to be ready"""
        for _ in range(max_wait // 2):
            try:
                description = self.pc.describe_index(index_name)
                status = getattr(description, "status", None)

                if isinstance(status, dict) and status.get("ready"):
                    print("Index is ready.")
                    return

                if getattr(status, "ready", False):
                    print("Index is ready.")
                    return

            except Exception:
                pass

            time.sleep(2)

        print(f"Index creation requested. If operations fail, wait and retry.")
    
    def get_index(self):
        """Get Pinecone index instance"""
        if self._index is None:
            self._index = self.pc.Index(self.config.vector_db.index_name)
        return self._index
    
    def clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata for Pinecone compatibility
        Pinecone metadata values must be simple types: string, number, boolean, or list of strings
        """
        cleaned = {}

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                cleaned[key] = [str(item) for item in value]
            else:
                cleaned[key] = str(value)

        return cleaned
    
    def upsert_documents(
        self,
        documents: List[Document],
        embeddings: OpenAIEmbeddings,
        namespace: str = None
    ) -> None:
        """
        Upsert documents to Pinecone index
        
        Args:
            documents: List of documents to upsert
            embeddings: Embedding model to use
            namespace: Optional namespace for isolation
        """
        if namespace is None:
            namespace = self.config.vector_db.namespace
        
        index = self.get_index()
        batch_size = self.config.document_processing.batch_size
        total = len(documents)

        for batch_start in range(0, total, batch_size):
            batch = documents[batch_start: batch_start + batch_size]
            texts = [doc.page_content for doc in batch]
            vectors = embeddings.embed_documents(texts)

            upsert_payload = []

            for doc, vector in zip(batch, vectors):
                chunk_id = doc.metadata["chunk_id"]
                metadata = self.clean_metadata({
                    **doc.metadata,
                    "text": doc.page_content,
                })

                upsert_payload.append({
                    "id": chunk_id,
                    "values": vector,
                    "metadata": metadata,
                })

            kwargs = {"vectors": upsert_payload}
            if namespace:
                kwargs["namespace"] = namespace

            index.upsert(**kwargs)

            print(f"Upserted {min(batch_start + batch_size, total)} / {total} chunks")

        print("Finished upserting documents to Pinecone.")
    
    def query_similar(
        self,
        query_vector: List[float],
        k: int = None,
        namespace: str = None,
        filter_dict: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Query similar vectors from Pinecone
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            namespace: Optional namespace
            filter_dict: Optional metadata filter
            
        Returns:
            List of matches from Pinecone
        """
        if k is None:
            k = self.config.retrieval.k
        
        if namespace is None:
            namespace = self.config.vector_db.namespace
        
        index = self.get_index()
        
        query_kwargs = {
            "vector": query_vector,
            "top_k": k,
            "include_metadata": True,
        }

        if namespace:
            query_kwargs["namespace"] = namespace

        if filter_dict:
            query_kwargs["filter"] = filter_dict

        results = index.query(**query_kwargs)

        # Extract matches
        matches = getattr(results, "matches", None)
        if matches is None and isinstance(results, dict):
            matches = results.get("matches", [])

        return matches or []