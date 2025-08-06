import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your custom modules
from models.llm import get_chatgroq_model
from models.embeddings import get_embedding_model
from utils.helpers import get_pdf_text, get_text_chunks, get_vector_store
from utils.tools import get_web_search_results


def get_chat_response(chat_model, messages, system_prompt=""):
    """Get response from the chat model"""
    try:
        # Prepare messages for the model, including a system prompt
        formatted_messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        
        # Get response from model
        response = chat_model.invoke(formatted_messages)
        return response.content
    
    except Exception as e:
        # Return a user-friendly error message
        return f"Error getting response: {str(e)}"

def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! This chatbot is enhanced with a RAG pipeline to answer questions about your documents.")
    # You can expand this page with more detailed instructions if you like.
    st.info("Navigate to the 'Chat' page to begin.")
    

def chat_page():
    """Main chat interface page"""
    st.title("🤖 AI Study Buddy")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        chat_mode = st.radio(
            "Select Chat Mode:",
            ("Document Chat", "Web Search")
        )

        # --- NEW: RESPONSE MODE SELECTOR ---
        response_mode = st.radio(
            "Select Response Mode:",
            ("Concise", "Detailed"),
            index=1 # Default to Detailed
        )
        st.divider()

        if chat_mode == "Document Chat":
            st.header("Upload Your Document")
            pdf_file = st.file_uploader("Upload your PDF study material", type="pdf")

            if pdf_file and "vector_store" not in st.session_state:
                with st.spinner("Processing PDF..."):
                    raw_text = get_pdf_text(pdf_file)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        embedding_model = get_embedding_model()
                        st.session_state.vector_store = get_vector_store(text_chunks, embedding_model)
                        st.sidebar.success("Document processed successfully!")
                    else:
                        st.sidebar.error("Could not extract text from the PDF.")

    # --- NEW: DYNAMIC SYSTEM PROMPT LOGIC ---
    base_prompt = "You are a friendly and encouraging AI Study Buddy. Your goal is to help users understand topics. Always be positive, patient, and helpful."
    if response_mode == "Concise":
        system_prompt = base_prompt + " Your responses must be short, summarized, and to the point, in 1-2 sentences."
    else: # Detailed
        system_prompt = base_prompt + " Your responses must be expanded, in-depth, and provide comprehensive explanations."

    # Initialize chat model and history
    chat_model = get_chatgroq_model()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input logic
   # Chat input logic
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ""
                # Logic to handle different chat modes
                if chat_mode == "Web Search":
                    search_results = get_web_search_results(query=prompt)
                    final_prompt = f"Based on the following web search results, please provide an answer to the user's question.\n\nSearch Results:\n---\n{search_results}\n---\n\nQuestion: {prompt}"
                    response = get_chat_response(chat_model, [{"role": "user", "content": final_prompt}], system_prompt)

                elif chat_mode == "Document Chat":
                    # If a document is uploaded and processed
                    if "vector_store" in st.session_state and st.session_state.vector_store:
                        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                        docs = retriever.get_relevant_documents(prompt)
                        context = "\n".join([doc.page_content for doc in docs])
                        
                        flexible_prompt = f"Use the following document excerpts to answer the user's question. If the question is conversational or seems unrelated to the excerpts, answer it from your own knowledge. \n\nContext Excerpts:\n---\n{context}\n---\n\nUser's Question: {prompt}"
                        
                        model_messages = list(st.session_state.messages)
                        model_messages[-1] = {"role": "user", "content": flexible_prompt}
                        response = get_chat_response(chat_model, model_messages, system_prompt)
                    
                    # --- THIS IS THE FIX ---
                    # If no document is uploaded yet, have a normal conversation
                    else:
                        response = get_chat_response(chat_model, st.session_state.messages, system_prompt)

                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        


def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="LangChain RAG ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Navigation
    with st.sidebar:
        st.title("Navigation")
        # Add a button to clear the vector store and start over
        if st.button("Clear Document & Chat", use_container_width=True):
            st.session_state.messages = []
            if "vector_store" in st.session_state:
                del st.session_state.vector_store
            st.rerun()
            
        st.divider()
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )
    
    # Page routing
    if page == "Instructions":
        instructions_page()
    else: # Default to chat page
        chat_page()


if __name__ == "__main__":
    main()