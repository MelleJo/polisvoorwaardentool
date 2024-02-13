import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Custom embedding function using the pre-loaded Sentence Transformer model
def custom_embedding_function(texts):
    return model.encode(texts, convert_to_tensor=False, show_progress_bar=False).tolist()

# Function to add documents to ChromaDB with custom embeddings
def add_document_to_chroma_custom_embedding(file_path, document_text):
    embeddings = custom_embedding_function([document_text])
    collection.add(embeddings=embeddings, documents=[document_text], ids=[file_path])

# Initialize ChromaDB client and attempt to create or get a collection
chroma_client = chromadb.Client()
collection_name = "Polisvoorwaardentool_embeddings"
try:
    collection = chroma_client.create_collection(name=collection_name)
except Exception:
    collection = chroma_client.get_collection(name=collection_name)

# Set the base directory to the folder containing your PDFs
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

# Verify and potentially correct BASE_DIR
if not os.path.exists(BASE_DIR):
    st.error(f"Directory does not exist: {BASE_DIR}")
else:
    # Function to get categories based on subdirectories in BASE_DIR
    def get_categories():
        return sorted(next(os.walk(BASE_DIR))[1])

    # Function to list documents within a selected category
    def get_documents(category):
        category_path = os.path.join(BASE_DIR, category)
        return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

    # Extract text from a given PDF file path
    def extract_text_from_pdf(file_path):
        document_text = ""
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    document_text += text + "\n"
        return document_text

    # Wrapper function to add documents to ChromaDB
    def add_document_to_chroma(file_path, document_text):
        add_document_to_chroma_custom_embedding(file_path, document_text)

    # Query ChromaDB for the most relevant document based on a given question
    def query_chroma(question):
        results = collection.query(query_texts=[question], n_results=1)
        if results and results[0]['matches']:
            return results[0]['matches'][0]['id']
        return None

    # Generate an answer from a document based on a given question
    def get_answer(document_id, question):
        document_text = extract_text_from_pdf(document_id)
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview")
        response = llm.generate(
            SystemMessage(content=document_text),
            HumanMessage(content=question),
        )
        return response.generations[0][0].text if response.generations else "No response generated."

    def main():
        st.title("Polisvoorwaardentool - test versie 1.1. - chromadb")
        categories = get_categories()
        selected_category = st.selectbox("Kies een categorie:", categories)
        documents = get_documents(selected_category)
        selected_document = st.selectbox("Selecteer een polisvoorwaardendocument:", documents)
        document_path = os.path.join(BASE_DIR, selected_category, selected_document)
        
        # Extract and add document text to ChromaDB
        document_text = extract_text_from_pdf(document_path)
        add_document_to_chroma(document_path, document_text)

        question = st.text_input("Vraag maar raak:")
        if st.button("Antwoord") and question:
            relevant_document_id = query_chroma(question)
            if relevant_document_id:
                answer = get_answer(relevant_document_id, question)
                st.write(answer)
            else:
                st.error("Geen relevante documenten gevonden.")

    if __name__ == "__main__":
        main()
