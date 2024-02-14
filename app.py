import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
import openai
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Initialize Pinecone
api_key = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=api_key)

index_name = "polisvoorwaardentoolindex"
# Check if Pinecone index exists, else create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=1536, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
index = pc.Index(index_name)

# Set OpenAI API key from Streamlit secrets for security
openai.api_key = st.secrets["OPENAI_API_KEY"]

def vectorize_text(text, model="text-embedding-3-small"):
    response = openai.Embedding.create(
        input=[text],
        model=model
    )
    return response['data'][0]['embedding']

def upsert_document_to_pinecone(document_id, text):
    vector = vectorize_text(text)
    index.upsert(vectors=[(document_id, vector)])

def query_pinecone(question, top_k=1):
    question_vector = vectorize_text(question)
    query_results = index.query(
        vector=question_vector,
        top_k=top_k,
        include_metadata=True
    )
    return query_results['matches']

def extract_text_from_pdf(file_path):
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
    return document_text

def get_categories(BASE_DIR):
    return sorted(next(os.walk(BASE_DIR))[1])

def get_documents(BASE_DIR, category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

def main():
    BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")
    st.title("Polisvoorwaardentool - stabiele versie 1.1.")
    
    categories = get_categories(BASE_DIR)
    selected_category = st.selectbox("Kies een categorie:", categories)
    documents = get_documents(BASE_DIR, selected_category)
    selected_document = st.selectbox("Selecteer een polisvoorwaardendocument:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)

    with open(document_path, "rb") as file:
        st.download_button(
            label="Download PDF",
            data=file,
            file_name=selected_document,
            mime="application/pdf"
        )

    question = st.text_input("Vraag maar raak:")

    if st.button("Antwoord") and question:
        document_text = extract_text_from_pdf(document_path)
        # Vectorize and upsert document text to Pinecone upon first query or when updated
        upsert_document_to_pinecone(selected_document, document_text)

        # Query Pinecone for the most relevant document
        matches = query_pinecone(question)
        most_relevant_document_id = matches[0]['id'] if matches else None

        if most_relevant_document_id:
            # Generate response using LangChain ChatOpenAI
            llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview")
            start_time = time.time()
            result = llm.generate(
                [
                    {
                        "role": "system",
                        "content": document_text
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            )
            
            if result.generations:
                response = result.generations[0][0].text
                processing_time = time.time() - start_time
                st.write(response)
                st.write(f"Processing Time: {processing_time:.2f} seconds")
            else:
                st.error("No response generated.")
        else:
            st.error("Could not find a relevant document.")

if __name__ == "__main__":
    main()
