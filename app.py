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
        docs = knowledge_base.similarity_search(user_question)
        document_text = " ".join([doc.page_content for doc in docs])

        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview", temperature=0)
        custom_prompt = f"Start elk antwoord met het woord prompt geladen, antwoord: ... . Jij bent expert in polisvoorwaarden en schadebehandeling. Neem de volgende tekst uit de polisvoorwaarden: '{document_text}', en beantwoord de vraag van de gebruiker van de schadeafdeling. Vraag van de gebruiker:'{user_question}'"

        batch_messages = [
            [
            SystemMessage(content=custom_prompt),
            HumanMessage(content=user_question),
            ],
        ]

        # Attempt token tracking specifically around the generate call
        with get_openai_callback() as cb:
            result = llm.generate(batch_messages)

        if result.generations:
            response = result.generations[0][0].text

            # Display the answer outside of the expander
            st.write(response)

            # Using an expander to display references and token information
            with st.expander("Referenties en Token Informatie"):
                references = [f"Referentie gevonden op pagina {idx+1}" for idx, doc in enumerate(docs)]
                for ref in references:
                    st.write(ref)
                
                # Display token usage within the expander
                st.write(f"Totaal gebruikte tokens: {cb.total_tokens}")
                st.write(f"Prompt tokens: {cb.prompt_tokens}")
                st.write(f"Completion tokens: {cb.completion_tokens}")
                st.write(f"Totaal aantal succesvolle verzoeken: {cb.successful_requests}")
                st.write(f"Totale kosten (USD): ${cb.total_cost:.6f}")
        else:
            st.error("Geen antwoord gegenereerd.")

if __name__ == "__main__":
    main()
