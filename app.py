import os
import streamlit as st
from llama_index import VectorStoreIndex, Document, ServiceContext
from llama_index.llms import OpenAI
from pathlib import Path
import PyPDF2
import openai

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"
st.header("Polisvoorwaardentool")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to categorize PDFs
def categorize_pdfs(pdf_list):
    category_map = {}
    for pdf in pdf_list:
        prefix = os.path.basename(pdf).split('_')[0]
        # Mapping logic based on prefixes
        category_map.setdefault(
            {
                "Auto": "Autoverzekering",
                "AVB": "Bedrijfsaansprakelijkheidsverzekering",
                "BestAVB": "AVB Bestuurders",
                "BS": "Bedrijfsschadeverzekering",
                "BestAuto": "Bestelautoverzekering",
                "Brand": "Brandverzekeringen",
                "Cara": "Caravanverzekering",
                "EigVerv": "Eigen vervoer",
                "Fiets": "Fietsverzekering",
                "Geb": "Gebouwen",
                "GW": "Goed Werkgeverschap",
                "IB": "Inboedelverzekering",
                "Inv": "Inventaris",
                "Mot": "Motorverzekering",
                "RB": "Rechtsbijstandverzekering",
                "Reis": "Reisverzekering",
                "Scoot": "Scootmobielverzekering",
                "WEGAS": "WEGAS",
                "WerkMat": "Werk- en landbouwmaterieelverzekering",
                "WEGAM": "Werkgeversaansprakelijkheid Motorrijtuigen (WEGAM)",
                "Woon": "Woonhuisverzekering"
            }.get(prefix, "Overige"), []
        ).append(pdf)
    return category_map

# Function to load and index a document
def load_and_index_document(document_path):
    with st.spinner("Loading and indexing the document..."):
        loader = PDFReader()
        documents = loader.load_data(file=Path(document_path))
        doc = Document(content=documents[0].content)
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="...")
        )
        index = VectorStoreIndex.from_documents([doc], service_context=service_context)
        return index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Specify the directory where your PDFs are stored
pdf_dir = "./polisvoorwaardentool/preloaded_pdfs/"

if not os.path.exists(pdf_dir):
    st.warning("PDF directory does not exist.")
    st.stop()

# Retrieve all PDFs from the directory
all_pdfs = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

if not all_pdfs:
    st.warning("No PDFs found in the directory.")
    st.stop()

# Categorize the PDFs
category_map = categorize_pdfs(all_pdfs)

# Category and Document Selection
category = st.selectbox("Choose a category", list(category_map.keys()))
document_name = st.selectbox("Choose a document", category_map[category])

if 'chat_engine' not in st.session_state or st.session_state.selected_document != document_name:
    st.session_state.chat_engine = load_and_index_document(os.path.join(pdf_dir, document_name))

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
