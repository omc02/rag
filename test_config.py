#!/usr/bin/env python3
"""
Configuration Test Script
Run this to verify your enhanced RAG pipeline setup
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_configuration():
    """Test configuration loading"""
    print("🔧 Testing Configuration Loading...")
    
    try:
        from src.config import load_config
        config = load_config()
        
        print("✅ Configuration loaded successfully!")
        print(f"   📊 Embedding Model: {config.models.embedding_model}")
        print(f"   🤖 Chat Model: {config.models.chat_model}")
        print(f"   🔗 Pinecone Index: {config.vector_db.index_name}")
        print(f"   📁 PDF Directory: {config.get_pdf_dir()}")
        print(f"   ⚙️  Chunk Size: {config.document_processing.chunk_size}")
        print(f"   🔍 Retrieval K: {config.retrieval.k}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_environment():
    """Test environment setup"""
    print("\n🌍 Testing Environment Setup...")
    
    try:
        import os
        from src.utils.helpers import validate_api_keys
        
        # Check for env file
        env_file = Path("env/api_keys.env")
        if env_file.exists():
            print("✅ Environment file found")
            
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv(env_file)
        else:
            print("⚠️  Environment file not found - checking system environment")
        
        # Validate API keys
        key_status = validate_api_keys(["OPENAI_API_KEY", "PINECONE_API_KEY"])
        
        for key, is_valid in key_status.items():
            if is_valid:
                print(f"   ✅ {key}: Found")
            else:
                print(f"   ❌ {key}: Missing")
        
        all_keys_valid = all(key_status.values())
        if all_keys_valid:
            print("✅ All API keys are configured")
        else:
            print("❌ Some API keys are missing")
        
        return all_keys_valid
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


def test_dependencies():
    """Test that all dependencies are installed"""
    print("\n📦 Testing Dependencies...")
    
    required_packages = [
        ("yaml", "PyYAML"),
        ("pinecone", "Pinecone"),
        ("openai", "OpenAI"),
        ("langchain", "LangChain"),
        ("streamlit", "Streamlit"),
        ("dotenv", "python-dotenv"),
        ("pydantic", "Pydantic")
    ]
    
    missing_packages = []
    
    for package, display_name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {display_name}")
        except ImportError:
            print(f"   ❌ {display_name} - Not installed")
            missing_packages.append(display_name)
    
    if not missing_packages:
        print("✅ All dependencies are installed")
        return True
    else:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install -r requirements_rag_fixed.txt")
        return False


def test_project_structure():
    """Test project structure"""
    print("\n📁 Testing Project Structure...")
    
    required_dirs = [
        "src",
        "src/config", 
        "src/models",
        "src/data", 
        "src/vector_db",
        "src/retrieval",
        "src/chains",
        "src/pipeline",
        "src/utils",
        "config",
        "data/external"
    ]
    
    required_files = [
        "config/rag_config.yaml",
        "src/config/config_loader.py",
        "src/pipeline/rag_pipeline.py"
    ]
    
    missing_items = []
    
    # Check directories
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ - Missing")
            missing_items.append(dir_path)
    
    # Check files
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - Missing")
            missing_items.append(file_path)
    
    if not missing_items:
        print("✅ Project structure is complete")
        return True
    else:
        print(f"❌ Missing items: {len(missing_items)}")
        return False


def test_pipeline():
    """Test pipeline creation"""
    print("\n🚀 Testing Pipeline Creation...")
    
    try:
        from src.pipeline.rag_pipeline import create_pipeline
        
        print("   Creating pipeline...")
        pipeline = create_pipeline()
        
        print("   ✅ Pipeline created successfully")
        print(f"   📝 Config loaded: {pipeline.config.models.chat_model}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Pipeline creation failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 RAG Pipeline Configuration Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Project Structure", test_project_structure), 
        ("Configuration", test_configuration),
        ("Environment", test_environment),
        ("Pipeline", test_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your enhanced RAG pipeline is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Add PDF files to data/external/")
        print("   2. Run: streamlit run streamlit_app_new.py")
        print("   3. Or use the enhanced notebook cells")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)