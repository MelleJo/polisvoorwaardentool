import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_openai import OpenAI, Embeddings
from langchain.chains import LLMChain
from langchain.document_loaders import LocalDocumentLoader
from langchain.retrievers import EmbeddingsRetriever
from langchain.llms import LangChainGPT

# Set the base directory for preloaded PDFs
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardentoolVA")

# Initialize LangChain components
openai_api = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
embeddings = Embeddings(model="text-embedding-ada-002")
document_loader = LocalDocumentLoader()
llm = LangChainGPT(openai_api=openai_api)

# Function to get categories from the base directory
def get_categories():
    return sorted(os.listdir(BASE_DIR))

# Function to get document names within a selected category
def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

# Function to extract text from a PDF document
def extract_text_from_pdf(file_path):
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
    return document_text

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
        document_text = extract_text_from_pdf(document_path)
        
        # Load and chunk document, then get embeddings
        loaded_document = document_loader.load(document_path)
        chunks = loaded_document.split_to_chunks(max_length=1024, overlap=128)
        chunk_embeddings = embeddings.embed_documents(chunks)

        # Find the most relevant chunks
        question_embedding = embeddings.embed_text(question)
        top_chunks = EmbeddingsRetriever.retrieve_most_relevant(chunk_embeddings, question_embedding, top_k=5)

        # Combine the text of the top chunks for the LLM context
        combined_context = "\n".join([chunk.text for chunk in top_chunks])
        
        # Prepare the LLM call
        response = llm.ask(question, context=combined_context, model_name=model_version)
        
        if response:
            st.write(response)
            if debug_mode:
                st.subheader("Debug Informatie")
                st.write(f"Vraag: {question}")
                if st.checkbox('Toon documenttekst'):
                    st.write(combined_context)
        else:
            st.error("Geen antwoord gegenereerd.")

if __name__ == "__main__":
    main()
