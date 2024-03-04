import streamlit as st
import os
import time
import pyperclip
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate



BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

company_name_mapping = {
    "nn": "Nationale Nederlanden",
    "asr": "a.s.r.",
    "nlg": "NLG Verzekeringen",
    "avero": "Avéro Achmea",
    "europeesche": "Europeesche Verzekeringen",
    "aig": "AIG",
    "allianz": "Allianz",
    "bikerpolis": "Bikerpolis",
    "das": "DAS",
    "guardian": "Guardian",
    "noordeloos": "Noordeloos",
    "reaal": "Reaal",
    "unigarant": "Unigarant",
}

def get_all_documents():
    all_docs = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.pdf'):
                path = os.path.join(root, file)
                all_docs.append({'title': file, 'path': path})
    return all_docs

def get_insurance_companies(all_documents):
    companies = set()
    for doc in all_documents:
        parts = doc['title'].split('_')
        if len(parts) >= 2:
            company_key = parts[1].lower()
            company_name = company_name_mapping.get(company_key, company_key.capitalize())
            companies.add(company_name)
    return sorted(companies)

def get_categories():
    try:
        return sorted(next(os.walk(BASE_DIR))[1])
    except StopIteration:
        st.error("Error accessing categories. Please check if the directory exists and is not empty.")
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
    with st.spinner('Processing your question...'):
        # Extract text from the document
        document_pages = extract_text_from_pdf_by_page(document_path)
        
        template = "Given the following text from the policy conditions: '{document_text}', answer the user's question: '{user_question}'"
        
        prompt = ChatPromptTemplate.from_template(template)

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(document_pages, embeddings)
        
        # Perform similarity search
        docs = knowledge_base.similarity_search(user_question)
        document_text = " ".join([doc.page_content for doc in docs])
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview", temperature=0, streaming=True)
        chain = prompt | llm | StrOutputParser() 
        return chain.stream({
            "document_text": document_text,
            "user_question": user_question,
        })
    
        
    
        # Prepare the prompt for the LLM
        #custom_prompt = f"Given the following text from the policy conditions: '{document_text}', answer the user's question: '{user_question}'"
        
        # Initialize the ChatOpenAI model for streaming
        
        
        # Stream the response from the LLM
        #def generate_stream():
            #stream = llm.generate_stream(custom_prompt)
            #for response in stream:
                #yield response.text
        
        # Use st.write_stream to display the LLM responses in the app
        #st.write_stream(generate_stream())


def display_search_results(search_results):
    if not search_results:
        st.write("No documents found.")
        return
    
    if isinstance(search_results[0], str):
        search_results = [{'title': filename, 'path': os.path.join(BASE_DIR, filename)} for filename in search_results]

    selected_title = st.selectbox("Search results:", [doc['title'] for doc in search_results])
    selected_document = next((doc for doc in search_results if doc['title'] == selected_title), None)
    
    if selected_document:
        user_question = st.text_input("Ask a question about your PDF after selection:")
        if user_question:
            process_document(selected_document['path'], user_question)
            st.write_stream(process_document)

        # Download button for the selected PDF file
        with open(selected_document['path'], "rb") as file:
            btn = st.download_button(
                label="Download PDF",
                data=file,
                file_name=selected_document['title'],
                mime="application/pdf"
            )


    

def main():
    st.title("Policy Conditions Tool - Version 1.1 (FAISS)")
    all_documents = get_all_documents()
    selection_method = st.radio("Choose your document selection method:", 
                                ['Search for a document', 'Select via category', 'Search by insurance company'])

    if selection_method == 'Search for a document':
        search_query = st.text_input("Search for a policy condition document:", "")
        if search_query:
            search_results = [doc for doc in all_documents if search_query.lower() in doc['title'].lower()]
            display_search_results(search_results)

    elif selection_method == 'Select via category':
        categories = get_categories()
        if categories:
            selected_category = st.selectbox("Choose a category:", categories)
            document_filenames = get_documents(selected_category)  # Returns filenames
            # Construct full paths for documents in the selected category
            documents_with_paths = [{'title': filename, 'path': os.path.join(BASE_DIR, selected_category, filename)} for filename in document_filenames]
            display_search_results(documents_with_paths)

    elif selection_method == 'Search by insurance company':
        companies = get_insurance_companies(all_documents)
        selected_company = st.selectbox("Select an insurance company:", companies)
        if selected_company:
            original_keys = [key for key, value in company_name_mapping.items() if value.lower() == selected_company.lower()]
            if not original_keys:
                original_keys = [selected_company.lower()]
            company_documents = [doc for doc in all_documents if any(key for key in original_keys if key in doc['title'].lower())]
            display_search_results(company_documents)

if __name__ == "__main__":
    main()
