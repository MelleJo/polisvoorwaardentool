import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
import chromadb
from sentence_transformers import SentenceTransformer

# Load a pre-defined model
model = SentenceTransformer('all-MiniLM-L6-v2')

def custom_embedding_function(texts):
    return model.encode(texts, convert_to_tensor=False, show_progress_bar=False).tolist()

# Use this custom embedding function when adding documents
def add_document_to_chroma_custom_embedding(file_path, document_text):
    embeddings = custom_embedding_function([document_text])
    collection.add(embeddings=embeddings, documents=[document_text], ids=[file_path])

# Initialize ChromaDB client and create/get a collection for policy documents.
chroma_client = chromadb.Client()
collection_name = "Polisvoorwaardentool_embeddings"
try:
    collection = chroma_client.create_collection(name=collection_name)
except Exception:
    collection = chroma_client.get_collection(name=collection_name)

BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

def get_categories():
    return sorted(next(os.walk(BASE_DIR))[1])

def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

def extract_text_from_pdf(file_path):
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
    return document_text

def add_document_to_chroma(file_path, document_text):
    add_document_to_chroma_custom_embedding(file_path, document_text)


def query_chroma(question):
    # Query ChromaDB for the most relevant document ID based on the question.
    results = collection.query_texts([question], n_results=1)
    if results:
        return results[0]['ids'][0]  # Return the ID of the most relevant document.
    return None

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
    
    # Extract and add document text to ChromaDB when selected.
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
