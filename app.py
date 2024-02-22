import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import AnalyzeDocumentChain

BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

def get_categories():
    try:
        return sorted(next(os.walk(BASE_DIR))[1])
    except StopIteration:
        st.error(f"Failed to access categories in {BASE_DIR}. Check if the directory exists and is not empty.")
        return []

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

def main():
    st.title("Polisvoorwaardentool - verbeterde versie met FAISS")

    categories = get_categories()
    if not categories:
        return  # Stop further execution if no categories found
    
    selected_category = st.selectbox("Kies een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een polisvoorwaardendocument:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)

    with open(document_path, "rb") as file:
        st.download_button(label="Download PDF", data=file, file_name=selected_document, mime="application/pdf")

    question = st.text_input("Vraag maar raak:")
    
    if st.button("Antwoord") and question:
        document_text = extract_text_from_pdf(document_path)
        
        # Note: For actual usage, consider pre-processing your PDFs and storing their embeddings in FAISS.
        # Dynamically processing and embedding documents on each query can be inefficient for larger documents.
        
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        # Initialize FAISS vector store - this should ideally be done outside this function for efficiency
        vector_store = FAISS(dimension=embeddings.embedding_dimension)
        document_chunks = [document_text]  # Simplified to use the whole text as a single chunk for accuracy
        vector_store.add_texts(document_chunks, embeddings)
        
        # Assume direct usage of the document text for answering the question due to accuracy concerns
        llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        analyze_document_chain = AnalyzeDocumentChain(llm=llm)
        response = analyze_document_chain.run(input_document=document_text, question=question)
        
        st.write(response)

if __name__ == "__main__":
    main()
