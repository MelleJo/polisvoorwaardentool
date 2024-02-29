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

def get_all_documents():
    all_docs = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.pdf'):
                path = os.path.join(root, file)
                all_docs.append({'title': file, 'path': path})
    return all_docs

def get_categories():
    try:
        return sorted(next(os.walk(BASE_DIR))[1])
    except StopIteration:
        st.error(f"Error accessing categories in {BASE_DIR}. Please check if the directory exists and is not empty.")
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

def process_document(document_path, user_question):
    document_pages = extract_text_from_pdf_by_page(document_path)
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(document_pages, embeddings)
    docs = knowledge_base.similarity_search(user_question)
    document_text = " ".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview", temperature=0)
    custom_prompt = f"Given the following text from the policy conditions: '{document_text}', answer the user's question. User's question: '{user_question}'"

    with get_openai_callback() as cb:
        result = llm.generate([[SystemMessage(content=custom_prompt), HumanMessage(content=user_question)]])

    if result.generations:
        response = result.generations[0][0].text
        st.write(response)
        with st.expander("References and Token Information"):
            st.write(f"Total used tokens: {cb.total_tokens}")
            st.write(f"Prompt tokens: {cb.prompt_tokens}")
            st.write(f"Completion tokens: {cb.completion_tokens}")
            st.write(f"Total successful requests: {cb.successful_requests}")
            st.write(f"Total cost (USD): ${cb.total_cost:.6f}")
    else:
        st.error("No answer generated.")

def main():
    st.title("Policy Conditions Tool - Version 1.1 (FAISS)")
    selection_method = st.radio("Choose your document selection method:", ['Search for a document', 'Select via category'])

    if selection_method == 'Search for a document':
        search_query = st.text_input("Search for a policy condition document:", "")
        if search_query:
            all_documents = get_all_documents()
            search_results = [doc for doc in all_documents if search_query.lower() in doc['title'].lower()]
            if search_results:
                selected_title = st.selectbox("Search results:", [doc['title'] for doc in search_results])
                selected_document = next((doc for doc in search_results if doc['title'] == selected_title), None)
                if selected_document:
                    user_question = st.text_input("Ask a question about your PDF after selection:")
                    if user_question:
                        process_document(selected_document['path'], user_question)
            else:
                st.write("No documents found matching the search query.")
    elif selection_method == 'Select via category':
        categories = get_categories()
        if categories:
            selected_category = st.selectbox("Choose a category:", categories)
            documents = get_documents(selected_category)
            if documents:
                selected_document = st.selectbox("Select a policy condition document:", documents)
                document_path = os.path.join(BASE_DIR, selected_category, selected_document)
                user_question = st.text_input("Ask a question about your PDF after selection:")
                if user_question:
                    process_document(document_path, user_question)

if __name__ == "__main__":
    main()