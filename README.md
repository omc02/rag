# 🤖 RAG AI Personal Finance Assistant

*A modern, configuration-driven RAG pipeline for intelligent financial advice*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-orange.svg)](https://pinecone.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)

## 🎯 Project Overview

This project implements a **state-of-the-art Retrieval-Augmented Generation (RAG) pipeline** designed to serve as an intelligent personal finance knowledge assistant. Built with a modern, modular architecture, it combines advanced vector search technology with OpenAI's language models to provide accurate, context-aware responses to financial questions.

### 🌟 What Makes This Special

- **🔧 Configuration-Driven**: All settings in YAML - no code changes needed
- **🏗️ Modular Architecture**: Professional software design patterns
- **⚡ One-Command Setup**: `pipeline.full_setup()` and you're ready
- **🛡️ Type Safety**: Pydantic validation prevents configuration errors
- **🔄 Backward Compatible**: Legacy code still works alongside new features

## 🚀 Key Features

### **Enhanced Architecture**
- **🏭 Factory Patterns**: Clean model and component creation
- **📦 Modular Design**: Organized in logical modules for easy maintenance
- **⚙️ YAML Configuration**: Centralized settings for all components
- **🧪 Built-in Testing**: Comprehensive test suite included

### **Advanced RAG Capabilities**
- **📄 Smart PDF Processing**: Automated extraction and intelligent chunking
- **🔍 Semantic Search**: OpenAI text-embedding-3-small (1536 dimensions)
- **🗄️ Vector Database**: Pinecone serverless for blazing-fast retrieval
- **🤖 GPT-4o Integration**: Latest OpenAI models with fine-tuned prompts
- **🔗 LangChain Powered**: Professional RAG chain implementation

### **User Experience**
- **🖥️ Modern Streamlit UI**: Clean, responsive interface
- **📓 Enhanced Notebooks**: Both legacy and modern approaches
- **📊 Real-time Monitoring**: Configuration display and system stats
- **🎯 Intelligent Responses**: Context-aware financial advice

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Python 3.8+ | Core runtime |
| **RAG Framework** | LangChain | Pipeline orchestration |
| **LLM** | OpenAI GPT-4o/4o-mini | Text generation |
| **Embeddings** | OpenAI text-embedding-3-small | Semantic search |
| **Vector DB** | Pinecone | Document storage & retrieval |
| **UI** | Streamlit | Web interface |
| **Config** | PyYAML + Pydantic | Settings management |
| **Environment** | python-dotenv | Secrets management |

## ⚡ Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-ai-pipeline

# Install dependencies
pip install -r requirements_rag_fixed.txt
```

### 2️⃣ Configuration

1. **Set up API keys** in `env/api_keys.env`:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   PINECONE_API_KEY=your_pinecone_key_here
   ```

2. **Customize settings** in `config/rag_config.yaml`:
   ```yaml
   models:
     embedding_model: "text-embedding-3-small"
     chat_model: "gpt-4o-mini"
     temperature: 0.1
   
   retrieval:
     k: 10  # Number of documents to retrieve
   
   document_processing:
     chunk_size: 1000
     chunk_overlap: 200
   ```

3. **Add your PDFs** to `data/external/`

### 3️⃣ Test Your Setup

```bash
python test_config.py
```

### 4️⃣ Launch the Application

**🎨 Streamlit Web App (Recommended)**
```bash
streamlit run streamlit_app.py
```

**📓 Jupyter Notebook**
```python
# In main/v2_main.ipynb, set:
USE_NEW_PIPELINE = True
# Then run the enhanced cells
```

**🐍 Python Script**
```python
from src.pipeline.rag_pipeline import create_pipeline

# One-command setup!
pipeline = create_pipeline()
pipeline.full_setup()

# Ask questions
response = pipeline.ask_question("What is value investing?")
print(response['answer'])
```

## 🚀 How to Run - Step by Step

### **Method 1: Streamlit Web Interface (Easiest)**

1. **Open terminal/command prompt** in the project directory
2. **Test your setup** (recommended first):
   ```bash
   python test_config.py
   ```
3. **Launch the app**:
   ```bash
   streamlit run streamlit_app.py
   ```
4. **Open your browser** to the URL shown (usually `http://localhost:8501`)
5. **Start asking questions** about personal finance!

### **Method 2: Jupyter Notebook (Interactive)**

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```
2. **Open** `main/v2_main.ipynb`
3. **Set the flag** in the first cell:
   ```python
   USE_NEW_PIPELINE = True
   ```
4. **Run all cells** to see both legacy and enhanced approaches

### **Method 3: Python Script (Programmatic)**

```python
# Create a simple script (e.g., quick_test.py)
from src.pipeline.rag_pipeline import create_pipeline

# Initialize
pipeline = create_pipeline()
pipeline.full_setup()

# Ask a question
response = pipeline.ask_question("What is value investing according to Warren Buffett?")
print("Answer:", response['answer'])
print("Sources:", [doc.metadata.get('source_file') for doc in response.get('sources', [])])
```

### **Troubleshooting Run Issues**

| Issue | Solution |
|-------|----------|
| **Module not found** | Run: `pip install -r requirements_rag_fixed.txt` |
| **API key errors** | Check `env/api_keys.env` file exists with valid keys |
| **No documents found** | Add PDF files to `data/external/` directory |
| **Config errors** | Run: `python test_config.py` to diagnose |
| **Port already in use** | Use: `streamlit run streamlit_app.py --server.port 8502` |

## 📁 Project Structure

```
rag-ai-pipeline/
├── 📂 config/
│   └── rag_config.yaml          # 🎯 Centralized configuration
├── 📂 src/                      # 🏗️ Modular source code
│   ├── 📂 config/
│   │   └── config_loader.py     # Configuration management
│   ├── 📂 models/
│   │   └── llm_factory.py       # LLM creation factory
│   ├── 📂 data/
│   │   ├── document_loader.py   # PDF processing
│   │   └── text_splitter.py     # Intelligent chunking
│   ├── 📂 vector_db/
│   │   └── pinecone_client.py   # Vector database client
│   ├── 📂 retrieval/
│   │   └── retrievers.py        # Custom retrievers
│   ├── 📂 chains/
│   │   └── rag_chain.py         # RAG chain logic
│   ├── 📂 pipeline/
│   │   └── rag_pipeline.py      # 🌟 Main orchestrator
│   └── 📂 utils/
│       ├── logger.py            # Logging utilities
│       └── helpers.py           # Helper functions
├── 📂 data/external/            # 📄 Your PDF documents
├── 📂 env/                      # 🔐 Environment variables
│   └── api_keys.env
├── 📂 main/
│   └── v2_main.ipynb            # 📓 Enhanced notebook
├── streamlit_app_new.py         # 🎨 Modern web interface
├── test_config.py               # 🧪 System tester
├── requirements_rag_fixed.txt   # 📦 Dependencies
└── README.md                    # 📖 This file
```

## 🎛️ Configuration Guide

The power of this system lies in its configurability. Here are key settings you can adjust:

### **Model Settings**
```yaml
models:
  embedding_model: "text-embedding-3-small"  # or text-embedding-3-large
  embedding_dimension: 1536
  chat_model: "gpt-4o-mini"                  # or gpt-4o for more power
  temperature: 0.1                           # 0.0-1.0 (creativity level)
  max_tokens: 1200                           # Response length limit
```

### **Retrieval Tuning**
```yaml
retrieval:
  k: 10                    # Documents to retrieve (5-20 recommended)
  score_threshold: 0.7     # Similarity threshold (0.0-1.0)
  
document_processing:
  chunk_size: 1000         # Character per chunk (500-2000)
  chunk_overlap: 200       # Overlap between chunks (100-300)
```

### **System Behavior**
```yaml
prompts:
  system_prompt: |
    You are a knowledgeable financial advisor...
    [Customize your system prompt here]
    
conversation:
  memory_length: 5         # Number of previous exchanges to remember
  
paths:
  pdf_directory: "data/external"  # Where to find your PDFs
```

## 🚀 Usage Examples

### **Basic Question Answering**
```python
from src.pipeline.rag_pipeline import create_pipeline

pipeline = create_pipeline()
pipeline.full_setup()

# Financial advice
response = pipeline.ask_question("How do I evaluate a stock using P/E ratios?")
print(response['answer'])
print(f"Sources: {response.get('sources', [])}")
```

### **Batch Processing**
```python
questions = [
    "What is diversification?",
    "How does compound interest work?", 
    "What are the risks of leverage?"
]

for question in questions:
    response = pipeline.ask_question(question)
    print(f"Q: {question}")
    print(f"A: {response['answer'][:200]}...")
    print("---")
```

### **Testing Retrieval**
```python
# See what documents are retrieved for a query
results = pipeline.test_retrieval("value investing", k=5)
for i, result in enumerate(results):
    print(f"{i+1}. {result['source']} (Score: {result['score']:.3f})")
    print(f"   {result['content'][:150]}...")
```

## 📊 Performance & Monitoring

The system includes built-in monitoring and optimization features:

### **System Stats**
- **Response Time**: Average query processing time
- **Document Coverage**: How many of your PDFs are being used
- **Retrieval Quality**: Similarity scores and source diversity
- **Token Usage**: OpenAI API consumption tracking

### **Optimization Tips**

| Issue | Solution |
|-------|----------|
| Slow responses | Reduce `k` value or chunk size |
| Poor answers | Increase `k` or lower score_threshold |
| Generic responses | Improve system prompt or add more documents |
| High API costs | Use gpt-4o-mini, reduce max_tokens |

## 🧪 Testing & Quality Assurance

### **Run the Test Suite**
```bash
# Full system test
python test_config.py

# Test specific components
python -m src.config.config_loader
python -m src.pipeline.rag_pipeline
```

### **Manual Testing Queries**
The system works best with financial questions like:
- "What is the difference between growth and value investing?"
- "How should I diversify my portfolio?"
- "What are the key financial ratios for stock analysis?"
- "Explain the concept of risk-adjusted returns"

## 🔧 Troubleshooting

### **Common Issues**

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError` | Missing dependencies | `pip install -r requirements_rag_fixed.txt` |
| API key errors | Invalid or missing keys | Check `env/api_keys.env` |
| No PDF documents found | Empty data directory | Add PDFs to `data/external/` |
| Configuration errors | Invalid YAML syntax | Run `python test_config.py` |
| Poor answer quality | Insufficient context | Increase `k` in config, add more documents |

### **Debug Mode**
Enable detailed logging by setting in your config:
```yaml
logging:
  level: DEBUG
  file: "rag_pipeline.log"
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow the modular structure**: Add new components in appropriate `src/` subdirectories
4. **Update configuration schema** if adding new settings
5. **Add tests** for new functionality
6. **Update documentation**
7. **Submit a pull request**

### **Development Setup**
```bash
# Development dependencies
pip install -r requirements_rag_fixed.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

## 🗺️ Roadmap

### **Upcoming Features**
- **🌐 Multi-language Support**: Process documents in different languages
- **🔄 Hybrid Search**: Combine semantic and keyword search
- **📈 Analytics Dashboard**: Advanced usage and performance metrics
- **🎯 Fine-tuning**: Custom model training on your documents
- **🔌 API Endpoint**: REST API for external integrations
- **📱 Mobile UI**: Responsive design for mobile devices

### **Integrations Planned**
- **📊 Google Sheets**: Direct export of insights
- **📧 Email Summaries**: Scheduled financial updates  
- **💬 Slack Bot**: Team-based financial Q&A
- **🔗 Web Scraping**: Live financial data ingestion

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT-4o and embedding models
- **Pinecone** for vector database infrastructure
- **LangChain** for RAG framework
- **Streamlit** for the beautiful web interface
- **The open-source community** for all the amazing tools

---

## 📞 Support

Having issues? Here's how to get help:

1. **📖 Check this README** for common solutions
2. **🧪 Run the test suite**: `python test_config.py`
3. **🐛 Check the logs** for detailed error information
4. **💬 Open an issue** on GitHub with:
   - Your configuration file (without API keys)
   - Error messages
   - Steps to reproduce

---

<div align="center">

**🚀 Ready to revolutionize your financial knowledge management?**

*Get started in under 5 minutes with our configuration-driven RAG pipeline!*

</div>
     chat_model: "gpt-4o-mini"
     temperature: 0.1
   
   retrieval:
     k: 10  # Number of documents to retrieve
   
   document_processing:
     chunk_size: 1000
     chunk_overlap: 200
   ```

3. **Add your PDFs** to `data/external/`

### 3️⃣ Test Your Setup

```bash
python test_config.py
```

### 4️⃣ Launch the Application

**🎨 Streamlit Web App (Recommended)**
```bash
streamlit run streamlit_app_new.py
```

**📓 Jupyter Notebook**
```python
# In main/v2_main.ipynb, set:
USE_NEW_PIPELINE = True
# Then run the enhanced cells
```

**🐍 Python Script**
```python
from src.pipeline.rag_pipeline import create_pipeline

# One-command setup!
pipeline = create_pipeline()
pipeline.full_setup()

# Ask questions
response = pipeline.ask_question("What is value investing?")
print(response['answer'])
```

## Project Structure

```
rag-ai-pipeline/
│
├── data/
│   └── external/          # PDF documents for ingestion
│
├── env/
│   └── api_keys.env       # Environment variables and API keys
│
├── main/
│   └── main.ipynb         # Main notebook with RAG pipeline implementation
│
└── src/
    └── prompt.py          # Additional source files (if any)
```

## Usage

### Running the Pipeline

Execute the `main.ipynb` notebook sequentially to:

1. **Environment Setup**: Load API keys from `env/api_keys.env`
2. **Document Loading**: Extract PDF documents from `data/external/` directory
3. **Text Processing**: Split documents into semantically meaningful chunks
4. **Embedding Generation**: Create vector embeddings using Hugging Face transformers
5. **Index Creation**: Initialize or connect to Pinecone index 'ai-bot'
6. **Document Upsert**: Store embeddings with metadata in Pinecone
7. **RAG Chain Setup**: Configure retriever and language model
8. **Query Testing**: Test the system with personal finance questions

### Example Queries

The system can accurately answer questions such as:
- "Who is Warren Buffett?"
- "What is a PE ratio and forward PE ratio?"
- "What is margin of safety?"
- "How do you value a stock?"
- "What are financial ratios?"
- "If I have $100,000, how should I invest in stocks?"
- "What do you mean by diversification?"

The assistant responds with "I don't know" when information is not available in the loaded documents, ensuring answer reliability.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Specify your license here]

## Acknowledgments

- OpenAI
- Pinecone
- LangChain
- Hugging Face