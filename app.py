import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import nltk
from nltk.tokenize import sent_tokenize

# Ensure you have the NLTK punkt tokenizer downloaded
nltk.download('punkt')

# Set the base directory for preloaded PDFs
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardentoolVA")

# Function to get categories from the base directory
def get_categories():
    return sorted(os.listdir(BASE_DIR))

# Function to get document names within a selected category
def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

# Function to extract text from a PDF document and chunk it
def extract_and_chunk_text(file_path, chunk_size=1024):
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
                
    # Tokenize the document text into sentences
    sentences = sent_tokenize(document_text)
    
    # Chunk the text into chunks of approximately chunk_size tokens
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk + " " + sentence) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    chunks.append(current_chunk)  # Add the last chunk
    
    return chunks

# Main Streamlit app function
def main():
    st.title("Polisvoorwaardentool - Testversie 1.0")
    
    # Debug mode toggle
    debug_mode = st.checkbox('Debugmodus', value=False)

    # Model version selection
    model_choice = st.selectbox("Kies model versie:", ["ChatGPT 3.5 Turbo", "gpt-4-turbo-preview"])
    model_version = "gpt-3.5-turbo" if model_choice == "ChatGPT 3.5 Turbo" else "gpt-4-turbo-preview"

    # Document selection UI
    categories = get_categories()
    selected_category = st.selectbox("Selecteer een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een document:", documents)
        
    # Question input
    question = st.text_input("Stel een vraag over het document:")
    
    if st.button("Krijg Antwoord") and question:
        document_path = os.path.join(BASE_DIR, selected_category, selected_document)
        chunks = extract_and_chunk_text(document_path)
        
        # Initialize ChatOpenAI with the selected model
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model=model_version)
        
        # For simplicity, use the first chunk as the context. Adjust as needed.
        context = chunks[0] if chunks else "Document is leeg of te kort."
        
        # Prepare the messages for the chat
        messages = [
            SystemMessage(content="Jij bent een expert in het analyseren van polisvoorwaarden. De gebruiker is een schadebehandelaar en wil graag jouw hulp bij het vinden van specifieke en relevante informatie voor de schadebehandeling van een polis. Nauwkeurigheid is prioriteit nummer 1"),
            SystemMessage(content=context),
            HumanMessage(content=question)
        ]
        
        # Get the response
        try:
            response = llm.invoke(messages)
            if response:
                st.write(response.content)
                if debug_mode:
                    st.subheader("Debug Informatie")
                    st.write(f"Vraag: {question}")
                    if st.checkbox('Toon documenttekst'):
                        st.write(context)  # Show the first chunk as context
            else:
                st.error("Geen antwoord gegenereerd.")
        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")

if __name__ == "__main__":
    main()
