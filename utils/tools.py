# utils/tools.py
import streamlit as st
from tavily import TavilyClient

# Try to get secrets from Streamlit, otherwise use local config
try:
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
except (KeyError, FileNotFoundError):
    from config.config import TAVILY_API_KEY

def get_web_search_results(query: str):
    """Uses the Tavily API to perform a web search."""
    try:
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily.qna_search(query=query, search_depth="basic")
        return response
    except Exception as e:
        return f"Error performing web search: {e}"