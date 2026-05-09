"""
LLM Factory for creating language model instances
Supports different providers (OpenAI, etc.)
"""

from typing import Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..config import RAGConfig


class LLMFactory:
    """Factory class for creating language model instances"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def create_embedding_model(self) -> OpenAIEmbeddings:
        """Create embedding model instance"""
        if self.config.models.embedding_provider.lower() == "openai":
            return OpenAIEmbeddings(
                model=self.config.models.embedding_model
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.config.models.embedding_provider}")
    
    def create_chat_model(self) -> ChatOpenAI:
        """Create chat model instance"""
        if self.config.models.chat_provider.lower() == "openai":
            return ChatOpenAI(
                model=self.config.models.chat_model,
                temperature=self.config.models.temperature,
                max_tokens=self.config.models.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported chat provider: {self.config.models.chat_provider}")
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the configured model"""
        return self.config.models.embedding_dimension