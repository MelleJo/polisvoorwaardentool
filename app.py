import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Define the path to your documents
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
    st.title("Polisvoorwaardentool - stabiele versie 1.0")

    debug_mode = st.checkbox('Toggle Debug Mode', value=False)
    model_choice = st.selectbox("Kies model versie:", ["ChatGPT 3.5 Turbo", "ChatGPT 4"])
    model_version = "gpt-3.5-turbo" if model_choice == "ChatGPT 3.5 Turbo" else "gpt-4-turbo-preview"

    categories = get_categories()
    selected_category = st.selectbox("Kies een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een polisvoorwaardendocument:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)
    document_text = extract_text_from_pdf(document_path)

    question = st.text_input("Vraag maar raak:")

    if st.button("Antwoord") and question:
        start_time = time.time()
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model=model_version)

        messages = [
            SystemMessage(content=document_text),
            HumanMessage(content=question),
        ]
        
        try:
            result = llm.generate(messages=messages)
            
            if result.generations:
                response = result.generations[0][0].text
                processing_time = time.time() - start_time

                st.write(response)

                if debug_mode:
                    st.subheader("Debug Information")
                    st.write(f"Vraag: {question}")
                    st.write(f"Verwerkingstijd: {processing_time:.2f} seconden")
                    st.write(f"Tokeengebruik: {result.llm_output['token_usage']}")
                    if st.checkbox('Toon documenttekst'):
                        st.write(document_text)
            else:
                st.error("Geen antwoord gegenereerd.")
        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")

if __name__ == "__main__":
    main()
