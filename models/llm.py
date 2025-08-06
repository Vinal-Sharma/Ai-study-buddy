# models/llm.py
import streamlit as st
from langchain_groq import ChatGroq

# Try to get secrets from Streamlit, otherwise use local config
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GROQ_MODEL_NAME = st.secrets["GROQ_MODEL_NAME"]
except (KeyError, FileNotFoundError):
    from config.config import GROQ_API_KEY, GROQ_MODEL_NAME

def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    try:
        groq_model = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")