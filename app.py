import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from pdfminer.high_level import extract_text
import os

openai.api_key = st.secrets.openai_key
st.header("Polisvoorwaardentool")

# Initialize the chat message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to convert a single PDF to text
def convert_pdf_to_text(pdf_path):
    return extract_text(pdf_path)

# Function to load and index a single document
def load_document(document_path):
    with st.spinner("Loading and indexing the document..."):
        text_content = convert_pdf_to_text(document_path)
        doc = Document(content=text_content)
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert in insurance policies. Please provide detailed and accurate responses."))
        index = VectorStoreIndex.from_documents([doc], service_context=service_context)
        return index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Category and Document Selection
categories = ["AVB Bestuurders", "AVB", "Autoverzekering", "..."] # Add all your categories here
category = st.selectbox("Choose a category", categories)
document_name = st.selectbox("Choose a document", os.listdir(f"./preloaded_pdfs/{category}"))  # Adjust the path as needed

# Load and index the selected document
if 'chat_engine' not in st.session_state or st.session_state.selected_document != document_name:
    st.session_state.chat_engine = load_document(f"./preloaded_pdfs/{category}/{document_name}")
    st.session_state.selected_document = document_name

# Chat interface
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate and display response
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})
