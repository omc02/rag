# Enhanced RAG AI Pipeline

## 🚀 What's New

This project has been enhanced with a **modular, configuration-driven architecture** that makes it easier to maintain, customize, and extend.

### Key Improvements

- ✅ **YAML Configuration**: All settings centralized in `config/rag_config.yaml`
- ✅ **Modular Architecture**: Organized code in `src/` directory with clear separation
- ✅ **Type Safety**: Configuration validation with Pydantic models
- ✅ **Better Error Handling**: Structured logging and error management
- ✅ **Enhanced Streamlit App**: Improved UI with configuration display
- ✅ **Pipeline Orchestration**: Single command to set up the entire system

## 📁 New Project Structure

```
rag-ai-pipeline/
├── config/
│   └── rag_config.yaml              # 🆕 Centralized configuration
├── src/                             # 🆕 Modular source code
│   ├── config/
│   │   ├── __init__.py
│   │   └── config_loader.py         # Configuration management
│   ├── models/
│   │   └── llm_factory.py          # LLM creation factory
│   ├── data/
│   │   ├── document_loader.py      # PDF loading
│   │   └── text_splitter.py        # Document chunking
│   ├── vector_db/
│   │   └── pinecone_client.py      # Pinecone operations
│   ├── retrieval/
│   │   └── retrievers.py           # Custom retrievers
│   ├── chains/
│   │   └── rag_chain.py            # RAG chain implementation
│   ├── pipeline/
│   │   └── rag_pipeline.py         # 🌟 Main orchestrator
│   └── utils/
│       └── logger.py               # Logging utilities
├── data/external/                   # Your PDF files
├── env/api_keys.env                # API keys
├── main/v2_main.ipynb              # 🔄 Enhanced notebook
├── streamlit_app_new.py            # 🆕 Enhanced Streamlit app
├── requirements_rag_fixed.txt      # 🔄 Complete dependencies
└── README_ENHANCED.md              # This file
```

## 🔧 Quick Start with Enhanced System

### 1. Install Dependencies

```bash
pip install -r requirements_rag_fixed.txt
```

### 2. Configure Your Settings

Edit `config/rag_config.yaml` to customize:
- Models (embedding/chat)
- Retrieval settings
- Vector database configuration
- Prompts and behavior

### 3. Set Up API Keys

Create `env/api_keys.env`:
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

### 4. Add Your Documents

Place PDF files in `data/external/`

### 5. Run with One Command

**Option A: Streamlit App (Recommended)**
```bash
streamlit run streamlit_app_new.py
```

**Option B: Jupyter Notebook**
- Open `main/v2_main.ipynb`
- Run the enhanced configuration cells
- Set `USE_NEW_PIPELINE = True`

**Option C: Python Script**
```python
from src.pipeline.rag_pipeline import create_pipeline

# Create and setup pipeline
pipeline = create_pipeline()
pipeline.full_setup()  # One command setup!

# Ask questions
response = pipeline.ask_question("What is value investing?")
print(response['answer'])
```

## 🎯 Configuration Examples

### Change Models
```yaml
# In config/rag_config.yaml
models:
  embedding:
    model: "text-embedding-3-large"  # More accurate
    dimension: 3072
  chat:
    model: "gpt-4o"                  # More powerful
    temperature: 0.2                 # More creative
```

### Adjust Retrieval
```yaml
retrieval:
  k: 15  # Retrieve more documents
  
document_processing:
  chunk_size: 1000    # Larger chunks
  chunk_overlap: 200  # More overlap
```

### Customize Prompts
```yaml
prompts:
  system_prompt: |
    You are a specialized financial advisor...
    [Your custom prompt here]
```

## 🔄 Migration from Legacy

If you have the old version:

1. **Keep your existing notebook** - it still works!
2. **Try the new system** - run the enhanced cells
3. **Gradually migrate** - use `USE_NEW_PIPELINE = True`

Both approaches coexist peacefully.

## 🛠️ Development

### Adding New Features

1. **New Models**: Add to `src/models/llm_factory.py`
2. **New Retrievers**: Add to `src/retrieval/retrievers.py`
3. **New Configurations**: Add to `config/rag_config.yaml` and `config_loader.py`

### Testing

```bash
# Run the pipeline test
python -m src.pipeline.rag_pipeline

# Test configuration loading
python -m src.config.config_loader
```

## 🎨 Advanced Features

### Custom Vector Database
Easily switch to Weaviate, Qdrant, etc. by:
1. Adding new client in `src/vector_db/`
2. Updating configuration
3. Creating new retriever

### Hybrid Search
Combine multiple retrieval methods:
```python
from src.retrieval.retrievers import HybridRetriever

hybrid = HybridRetriever([retriever1, retriever2], weights=[0.7, 0.3])
```

### Caching
Enable caching in configuration:
```yaml
performance:
  enable_caching: true
```

## 🐛 Troubleshooting

### "Module not found" errors
- Make sure you're in the project root directory
- Check that `src/` directory contains all modules
- Verify Python path is set correctly

### Configuration errors
- Validate your YAML syntax
- Check required sections exist
- Use the configuration test: `python -m src.config.config_loader`

### API key issues
- Verify `env/api_keys.env` exists and has correct keys
- Check file permissions

## 🤝 Contributing

1. **Add features to modular structure**
2. **Update configuration schema**
3. **Add tests**
4. **Update this README**

## 📊 Performance Benefits

| Aspect | Legacy | Enhanced |
|--------|--------|----------|
| Setup Time | Manual, error-prone | One command |
| Configuration | Scattered in code | Centralized YAML |
| Maintenance | Difficult | Easy |
| Testing | Manual | Automated |
| Extensibility | Limited | High |

---

🎉 **The enhanced system maintains 100% backward compatibility while providing a modern, scalable architecture for your RAG pipeline!**