"""
Configuration loader for RAG AI Pipeline
Handles loading and validation of YAML configuration files
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    chat_provider: str
    chat_model: str
    temperature: float
    max_tokens: int


@dataclass
class VectorDBConfig:
    """Configuration for vector database"""
    provider: str
    index_name: str
    cloud: str
    region: str
    namespace: str
    metric: str


@dataclass
class DocumentProcessingConfig:
    """Configuration for document processing"""
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    separators: list


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval"""
    k: int
    include_metadata: bool


@dataclass
class ConversationConfig:
    """Configuration for conversation management"""
    max_history: int
    show_sources: bool


@dataclass
class PathsConfig:
    """Configuration for file paths"""
    pdf_dir: str
    env_file: str
    config_dir: str
    src_dir: str


@dataclass
class PromptsConfig:
    """Configuration for system prompts"""
    system_prompt: str
    document_prompt_template: str


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str
    format: str
    file_enabled: bool
    file_path: str


@dataclass
class PerformanceConfig:
    """Configuration for performance settings"""
    max_retries: int
    retry_delay: int
    timeout: int
    enable_caching: bool


@dataclass
class UIConfig:
    """Configuration for UI settings"""
    title: str
    sidebar_title: str
    max_input_length: int
    default_question: str


class RAGConfig:
    """Main configuration class that holds all configuration sections"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path: Path to configuration YAML file. If None, will search for default locations.
        """
        if config_path is None:
            config_path = self._find_config_file()
        
        self._config_path = Path(config_path)
        self._raw_config = self._load_config()
        self._validate_config()
        self._parse_config()
    
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        possible_paths = [
            Path.cwd() / "config" / "rag_config.yaml",
            Path.cwd() / "rag_config.yaml",
            Path.cwd().parent / "config" / "rag_config.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(
            "Configuration file not found. Please create config/rag_config.yaml or specify config_path"
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(self._config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {self._config_path}: {e}")
    
    def _validate_config(self):
        """Validate required configuration sections exist"""
        required_sections = ['models', 'vector_db', 'document_processing', 'retrieval', 'conversation', 'paths', 'prompts']
        
        for section in required_sections:
            if section not in self._raw_config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _parse_config(self):
        """Parse configuration into typed dataclasses"""
        # Models configuration
        models_cfg = self._raw_config['models']
        self.models = ModelConfig(
            embedding_provider=models_cfg['embedding']['provider'],
            embedding_model=models_cfg['embedding']['model'],
            embedding_dimension=models_cfg['embedding']['dimension'],
            chat_provider=models_cfg['chat']['provider'],
            chat_model=models_cfg['chat']['model'],
            temperature=models_cfg['chat']['temperature'],
            max_tokens=models_cfg['chat']['max_tokens']
        )
        
        # Vector DB configuration
        vdb_cfg = self._raw_config['vector_db']
        self.vector_db = VectorDBConfig(
            provider=vdb_cfg['provider'],
            index_name=vdb_cfg['index_name'],
            cloud=vdb_cfg['cloud'],
            region=vdb_cfg['region'],
            namespace=vdb_cfg['namespace'],
            metric=vdb_cfg['metric']
        )
        
        # Document processing configuration
        doc_cfg = self._raw_config['document_processing']
        self.document_processing = DocumentProcessingConfig(
            chunk_size=doc_cfg['chunk_size'],
            chunk_overlap=doc_cfg['chunk_overlap'],
            batch_size=doc_cfg['batch_size'],
            separators=doc_cfg['separators']
        )
        
        # Retrieval configuration
        ret_cfg = self._raw_config['retrieval']
        self.retrieval = RetrievalConfig(
            k=ret_cfg['k'],
            include_metadata=ret_cfg['include_metadata']
        )
        
        # Conversation configuration
        conv_cfg = self._raw_config['conversation']
        self.conversation = ConversationConfig(
            max_history=conv_cfg['max_history'],
            show_sources=conv_cfg['show_sources']
        )
        
        # Paths configuration
        paths_cfg = self._raw_config['paths']
        self.paths = PathsConfig(
            pdf_dir=paths_cfg['pdf_dir'],
            env_file=paths_cfg['env_file'],
            config_dir=paths_cfg['config_dir'],
            src_dir=paths_cfg['src_dir']
        )
        
        # Prompts configuration
        prompts_cfg = self._raw_config['prompts']
        self.prompts = PromptsConfig(
            system_prompt=prompts_cfg['system_prompt'],
            document_prompt_template=prompts_cfg['document_prompt_template']
        )
        
        # Optional configurations with defaults
        self.logging = LoggingConfig(
            level=self._raw_config.get('logging', {}).get('level', 'INFO'),
            format=self._raw_config.get('logging', {}).get('format', '%(asctime)s - %(levelname)s - %(message)s'),
            file_enabled=self._raw_config.get('logging', {}).get('file_enabled', False),
            file_path=self._raw_config.get('logging', {}).get('file_path', 'logs/rag_pipeline.log')
        )
        
        self.performance = PerformanceConfig(
            max_retries=self._raw_config.get('performance', {}).get('max_retries', 3),
            retry_delay=self._raw_config.get('performance', {}).get('retry_delay', 2),
            timeout=self._raw_config.get('performance', {}).get('timeout', 30),
            enable_caching=self._raw_config.get('performance', {}).get('enable_caching', False)
        )
        
        self.ui = UIConfig(
            title=self._raw_config.get('ui', {}).get('title', 'RAG Assistant'),
            sidebar_title=self._raw_config.get('ui', {}).get('sidebar_title', 'Configuration'),
            max_input_length=self._raw_config.get('ui', {}).get('max_input_length', 500),
            default_question=self._raw_config.get('ui', {}).get('default_question', '')
        )
    
    def get_project_root(self) -> Path:
        """Get the project root directory"""
        # Start from config file location and go up to find project root
        current = self._config_path.parent.parent
        
        # Look for indicators of project root (data/, env/, etc.)
        while current != current.parent:
            if (current / "data").exists() or (current / "env").exists():
                return current
            current = current.parent
        
        # Fallback to config file's grandparent directory
        return self._config_path.parent.parent
    
    def get_pdf_dir(self) -> Path:
        """Get full path to PDF directory"""
        return self.get_project_root() / self.paths.pdf_dir
    
    def get_env_file(self) -> Path:
        """Get full path to environment file"""
        return self.get_project_root() / self.paths.env_file
    
    def reload(self):
        """Reload configuration from file"""
        self._raw_config = self._load_config()
        self._validate_config()
        self._parse_config()
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"RAGConfig loaded from: {self._config_path}"
    
    def __repr__(self) -> str:
        return self.__str__()


def load_config(config_path: Optional[str] = None) -> RAGConfig:
    """
    Convenience function to load configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        RAGConfig instance
    """
    return RAGConfig(config_path)


# Example usage
if __name__ == "__main__":
    # Test loading configuration
    try:
        config = load_config()
        print(f"✅ Configuration loaded successfully from: {config._config_path}")
        print(f"📊 Embedding model: {config.models.embedding_model}")
        print(f"🤖 Chat model: {config.models.chat_model}")
        print(f"📁 PDF directory: {config.get_pdf_dir()}")
        print(f"🔗 Pinecone index: {config.vector_db.index_name}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")