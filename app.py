import os
import streamlit as st
from llama_index import VectorStoreIndex, Document, ServiceContext
from llama_index.llms import OpenAI
from PyPDF2 import PdfReader
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
        category = {
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
        }.get(prefix, "Overige")
        category_map.setdefault(category, []).append(pdf)
    return category_map

# Function to load and index a document
def load_and_index_document(document_path):
    with st.spinner("Loading and indexing the document..."):
        try:
            with open(document_path, 'rb') as file:
                reader = PdfReader(file)
                document_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        document_text += text

            doc = Document(content=document_text)
            service_context = ServiceContext.from_defaults(
                llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="...")
            )
            index = VectorStoreIndex.from_documents([doc], service_context=service_context)
            return index.as_chat_engine(chat_mode="condense_question", verbose=True)
        except FileNotFoundError as e:
            st.error(f"File not found: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Specify the directory where your PDFs are stored
pdf_dir = "./preloaded_pdfs/"

if not os.path.exists(pdf_dir):
    st.warning("PDF directory does not exist.")
    st.stop()

# Retrieve all PDFs from the directory and its subdirectories
all_pdfs = []
for root, dirs, files in os.walk(pdf_dir):
    for file in files:
        if file.endswith('.pdf'):
            all_pdfs.append(os.path.join(root, file))

if not all_pdfs:
    st.warning("No PDFs found in the directory.")
    st.stop()

# Category and Document Selection
category = st.selectbox("Choose a category", list(category_map.keys()))
document_name = st.selectbox("Choose a document", category_map[category])

if 'chat_engine' not in st.session_state or st.session_state.selected_document != document_name:
    full_document_path = os.path.join(pdf_dir, document_name)  # Construct the full path here
    st.session_state.chat_engine = load_and_index_document(full_document_path)

st.session_state.selected_document = document_name


# Chat interface
if prompt := st.text_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.container():
        st.write(message["content"])  # Removed the 'key' argument

# Generate and display response
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    # Ensure chat_engine is not None before calling chat
    if st.session_state.get('chat_engine'):
        response = st.session_state.chat_engine.chat(st.session_state.messages[-1]["content"])
        st.session_state.messages.append({"role": "assistant", "content": response.response})
    else:
        st.error("Chat engine not initialized. Please select a document first.")
