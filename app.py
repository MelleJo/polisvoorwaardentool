import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback

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
        return
    
    selected_category = st.selectbox("Kies een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een polisvoorwaardendocument:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)

    with open(document_path, "rb") as file:
        st.download_button(label="Download PDF", data=file, file_name=selected_document, mime="application/pdf")

    document_pages = extract_text_from_pdf_by_page(document_path)

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(document_pages, embeddings)

    user_question = st.text_input("Stel een vraag over uw PDF:")
    
    if user_question:
        with get_openai_callback() as cb:
            docs = knowledge_base.similarity_search(user_question)
            document_text = " ".join([doc.page_content for doc in docs])
            references = [f"Referentie gevonden op pagina {idx+1}" for idx, doc in enumerate(docs)]

            for ref in references:
                st.write(ref)

            llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview", temperature=0)

            batch_messages = [
                [
                SystemMessage(content=document_text),
                HumanMessage(content=user_question),
                ],
            ]

            result = llm.generate(batch_messages)

            if result.generations:
                response = result.generations[0][0].text
                st.write(response)
            else:
                st.error("Geen antwoord gegenereerd.")

        # Display token usage
        st.write(f"Totaal gebruikte tokens: {cb.total_tokens}")
        st.write(f"Prompt tokens: {cb.prompt_tokens}")
        st.write(f"Completion tokens: {cb.completion_tokens}")
        st.write(f"Totaal aantal succesvolle verzoeken: {cb.successful_requests}")
        st.write(f"Totale kosten (USD): ${cb.total_cost:.6f}")

if __name__ == "__main__":
    main()
