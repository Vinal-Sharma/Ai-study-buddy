# models/embeddings.py

from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    """Initialize and return the HuggingFace embedding model."""
    try:
        # We will use a popular and efficient open-source embedding model.
        # By not specifying the device, the library will automatically use the CPU.
        model_name = "all-MiniLM-L6-v2"
        
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        
        return embedding_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")