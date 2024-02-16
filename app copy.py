import streamlit as st
import os
from PyPDF2 import PdfReader
import openai
from langchain.llms import OpenAI
from langchain.chains import SimpleQAChain
from pinecone import Pinecone, PodSpec
import numpy as np

# Set your API keys here
PINECONE_API_KEY = "your_pinecone_api_key"
OPENAI_API_KEY = "your_openai_api_key"

# Initialize Pinecone
Pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
index_name = "polisvoorwaardentoolindex"
if index_name not in Pinecone.list_indexes():
    Pinecone.create_index(name=index_name, dimension=768, pod_type="s1")
index = Pinecone.Index(index_name=index_name)

# Initialize OpenAI and LangChain
openai.api_key = OPENAI_API_KEY
llm = OpenAI(api_key=OPENAI_API_KEY)
qa_chain = SimpleQAChain(llm=llm)

def vectorize_text(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return np.array(response['data'][0]['embedding'])

def upsert_document_to_pinecone(document_id, text):
    vector = vectorize_text(text)
    index.upsert(vectors={document_id: vector})

def extract_text_from_pdf(file_path):
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
    return document_text

BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

def get_categories(base_dir):
    return sorted(next(os.walk(base_dir))[1])

def get_documents(base_dir, category):
    category_path = os.path.join(base_dir, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

def main():
    st.title("Polisvoorwaardentool - testversie")
    
    categories = get_categories(BASE_DIR)
    selected_category = st.selectbox("Kies een categorie:", categories)
    
    documents = get_documents(BASE_DIR, selected_category)
    selected_document = st.selectbox("Selecteer een polisvoorwaardendocument:", documents)
    
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)
    question = st.text_input("Vraag maar raak:")

    if st.button("Antwoord") and question:
        document_text = extract_text_from_pdf(document_path)
        upsert_document_to_pinecone(selected_document, document_text)
        response = qa_chain.run(document=document_text, question=question)
        st.write(response)

if __name__ == "__main__":
    main()
