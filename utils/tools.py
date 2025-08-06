# utils/tools.py

from tavily import TavilyClient
from config.config import TAVILY_API_KEY

def get_web_search_results(query: str):
    """
    Uses the Tavily API to perform a web search.
    """
    try:
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        # Perform a search and return the answer.
        response = tavily.qna_search(query=query, search_depth="basic")
        return response
    except Exception as e:
        return f"Error performing web search: {e}"