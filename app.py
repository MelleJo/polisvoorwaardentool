import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Adjust the base directory according to your setup
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
    st.title("Polisvoorwaardentool - Testversie 1.0")
    debug_mode = st.checkbox('Debugmodus', value=False)

    model_choice = st.selectbox("Kies model versie:", ["ChatGPT 3.5 Turbo", "ChatGPT 4"])
    model_version = "gpt-3.5-turbo" if model_choice == "ChatGPT 3.5 Turbo" else "gpt-4"

    categories = get_categories()
    selected_category = st.selectbox("Selecteer een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een document:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)
    document_text = extract_text_from_pdf(document_path)

    question = st.text_input("Stel een vraag over het document:")

    if st.button("Krijg Antwoord") and question:
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model=model_version)

        messages = [
            SystemMessage(content=document_text),
            HumanMessage(content=question)
        ]

        try:
            response = llm.invoke(messages=messages)
            if response:
                st.write(response[-1].content)  # Assuming the last message is the AI's response
                if debug_mode:
                    st.subheader("Debug Informatie")
                    st.write(f"Vraag: {question}")
                    if st.checkbox('Toon documenttekst'):
                        st.write(document_text)
            else:
                st.error("Geen antwoord gegenereerd.")
        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")

if __name__ == "__main__":
    main()
