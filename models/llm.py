# models/llm.py
# models/llm.py

import os
import sys
from langchain_groq import ChatGroq

# This line is the most important part - it imports your key and model
from config.config import GROQ_API_KEY, GROQ_MODEL_NAME

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    try:
        # This now uses your key and model from the config file
        groq_model = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")