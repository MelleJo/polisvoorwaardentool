import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Define the base directory for preloaded PDFs
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

# Function to get categories
def get_categories():
    return sorted(next(os.walk(BASE_DIR))[1])

# Function to get documents within a category
def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
    return document_text

# Main function to run the Streamlit app
def main():
    st.title("Polisvoorwaardentool - testversie 1.0")

    # Toggle for debug mode
    st.session_state.debug_mode = st.checkbox('Debugmodus', value=False)

    # Model version selection
    model_choice = st.selectbox("Kies model versie:", ["ChatGPT 3.5 Turbo", "ChatGPT 4"])
    model_version = "gpt-3.5-turbo" if model_choice == "ChatGPT 3.5 Turbo" else "gpt-4"

    # Category and document selection
    categories = get_categories()
    selected_category = st.selectbox("Selecteer een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een document:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)
    document_text = extract_text_from_pdf(document_path)

    # User input for question
    question = st.text_input("Stel een vraag over het document:")
    if question:
        # Processing with selected model
        start_time = time.time()
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model=model_version)

        # Prepare messages for the chat model
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=question)
        ]

        # Invoke the model with messages
        try:
            response = llm.invoke(messages=messages)
            processing_time = time.time() - start_time

            # Display response and debug information if debug mode is active
            if response:
                st.write(response.content)
                if st.session_state.debug_mode:
                    debug_information(processing_time, question, document_text, response.content)
            else:
                st.error("Geen antwoord gegenereerd.")
        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")

# Function to display debug information
def debug_information(processing_time, question, document_text, response_text):
    st.subheader("Debug Informatie")
    st.write(f"Vraag: {question}")
    st.write(f"Verwerkingstijd: {processing_time:.2f} seconden")
    if st.checkbox('Toon documenttekst'):
        st.write(document_text)
    st.write(f"Antwoord: {response_text}")

if __name__ == "__main__":
    main()
