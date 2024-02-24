import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

def get_categories():
    try:
        return sorted(next(os.walk(BASE_DIR))[1])
    except StopIteration:
        st.error(f"Fout bij toegang tot categorieën in {BASE_DIR}. Controleer of de map bestaat en niet leeg is.")
        return []

def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

def extract_text_from_pdf_by_page(file_path):
    pages_text = []
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return pages_text

def main():
    st.title("Polisvoorwaardentool - Testversie 1.1 (FAISS)")

    categories = get_categories()
    if not categories:
        return  # Stop verdere uitvoering als er geen categorieën gevonden zijn
    
    selected_category = st.selectbox("Kies een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een polisvoorwaardendocument:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)

    pdf = selected_document

    with open(document_path, "rb") as file:
        st.download_button(label="Download PDF", data=file, file_name=selected_document, mime="application/pdf")

    # Tekst uit elke pagina van de PDF extraheren
    document_pages = extract_text_from_pdf_by_page(document_path)

    # Maak embeddings voor elke pagina, aannemend dat je FAISS dienovereenkomstig aanpast
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(document_pages, embeddings)

    # Toon gebruikersinvoer
    user_question = st.text_input("Stel een vraag over uw PDF:")
    
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        document_text = " ".join([doc.page_content for doc in docs])
        references = [f"Referentie gevonden op pagina {idx+1}" for idx, doc in enumerate(docs)]

        # Referenties weergeven voor gebruikersinformatie (optioneel, voor debuggen)
        for ref in references:
            st.write(ref)
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview", temperature=0)

        batch_messages = [
            [
            SystemMessage(content=document_text),
            HumanMessage(content=user_question),
            ],
        ]

        try:
            result = llm.generate(batch_messages)
            
            # Het eerste antwoord uit het resultaat halen
            if result.generations:
                response = result.generations[0][0].text  # Aannemend dat de eerste generatie van de eerste batch is wat we willen
                
                st.write(response)  # Het antwoord weergeven
        
            else:
                st.error("Geen antwoord gegenereerd.")
        except Exception as e:
            st.error(f"Een fout is opgetreden: {e}")
        
if __name__ == "__main__":
    main()
