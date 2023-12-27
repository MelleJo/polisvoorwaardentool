import os
import streamlit as st

# PDF handling
from PyPDF2 import PdfReader

# LangChain specific imports
from langchain.chains import AnalyzeDocumentChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.question_answering import load_qa_chain

# Security
from hashlib import sha256

# Web app setup

# Set page title and other Streamlit configurations
st.set_page_config(page_title="VA-Polisvoorwaardentool")

# Password protection
hashed_password = st.secrets["hashed_password"]
password_input = st.text_input("Wachtwoord:", type="password")

# Password verification
if sha256(password_input.encode()).hexdigest() != hashed_password:
    st.error("Voer het juiste wachtwoord in.")
    st.stop()

# Global variable to cache embeddings to reduce repeated API calls
knowledge_bases = {}

# Function to categorize PDFs based on their filename
# Function to categorize PDFs based on their filename
def categorize_pdfs(pdf_list):
    category_map = {}
    for pdf in pdf_list:
        prefix = os.path.basename(pdf).split('_')[0]
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



def load_qa_chain(llm, chain_type="map_reduce"):
    # Setup for a QA Chain, adjust as needed
    if chain_type == "map_reduce":
        qa_chain = AnalyzeDocumentChain(llm)  # Assuming this is a valid initialization
        return qa_chain
    else:
        # Add other chain types if necessary
        raise ValueError(f"Unsupported chain type: {chain_type}")



# Main function where the app logic resides
def main():
    st.header("VA-Polisvoorwaardentool")

    # Set API key for OpenAI functionalities
    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key

    # Load the PDFs and categorize them
    pdf_dir = "preloaded_pdfs/"  # Adjust the path as needed
    all_pdfs = [os.path.join(dp, f) for dp, dn, filenames in os.walk(pdf_dir) for f in filenames if f.endswith('.pdf')]
    category_map = categorize_pdfs(all_pdfs)

    # Let the user select a category and a specific PDF
    categories = list(category_map.keys())
    selected_category = st.selectbox("Kies een categorie:", categories)
    available_pdfs = category_map[selected_category]
    pdf_names = [os.path.basename(pdf) for pdf in available_pdfs]
    selected_pdf_name = st.selectbox("Welke polisvoorwaarden wil je raadplegen?", pdf_names)

    # Process the selected PDF
    selected_pdf_index = pdf_names.index(selected_pdf_name)
    selected_pdf_path = available_pdfs[selected_pdf_index]
    pdf_text = process_pdf(selected_pdf_path)

    # Set up LangChain components
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.2)
    qa_chain = load_qa_chain(llm, chain_type="map_reduce")
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

    # Get user input and generate response
    user_question = st.text_input("Stel een vraag over de polisvoorwaarden")
    if user_question:
        response = qa_document_chain.run(
            input_document=pdf_text,
            question=user_question
        )
        st.write(response)

    # Helper function to process the selected PDF
    def process_pdf(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text

# Execute the main function when the script is run
if __name__ == '__main__':
    main()

