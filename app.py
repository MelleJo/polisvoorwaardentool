import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI

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

def main():
    st.title("Polisvoorwaardentool - testversie 1.0")
    st.session_state.debug_mode = st.checkbox('Debugmodus', value=False)

    model_choice = st.selectbox("Kies model versie:", ["ChatGPT 3.5 Turbo", "ChatGPT 4"])
    model_version = "gpt-3.5-turbo" if model_choice == "ChatGPT 3.5 Turbo" else "gpt-4"

    categories = get_categories()
    selected_category = st.selectbox("Selecteer een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een document:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)
    document_text = extract_text_from_pdf(document_path)

    question = st.text_input("Stel een vraag over het document:")
    if question:
        start_time = time.time()
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model=model_version)

        prompt = f"{document_text}\n\nQuestion: {question}"
        try:
            response = llm.invoke({"prompt": prompt, "max_tokens": 512})
            if response:
                processing_time = time.time() - start_time
                st.write(response["content"])
                if st.session_state.debug_mode:
                    debug_information(processing_time, question, document_text, response["content"])
            else:
                st.error("Geen antwoord gegenereerd.")
        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")

def debug_information(processing_time, question, document_text, response_text):
    st.subheader("Debug Informatie")
    st.write(f"Vraag: {question}")
    st.write(f"Verwerkingstijd: {processing_time:.2f} seconden")
    if st.checkbox('Toon documenttekst'):
        st.write(document_text)
    st.write(f"Antwoord: {response_text}")

if __name__ == "__main__":
    main()
