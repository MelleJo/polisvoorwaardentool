import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from hashlib import sha256

# Set page config
st.set_page_config(page_title="VA-Polisvoorwaardentool")

# Check password
hashed_password = st.secrets["hashed_password"]
password_input = st.text_input("Enter the password:", type="password")

if sha256(password_input.encode()).hexdigest() == hashed_password:
    st.success("Password correct!")
else:
    st.error("Password incorrect. Please try again.")
    st.stop()
    
# Function to display PDF using base64 encoding
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Global variable to cache embeddings to reduce repeated API calls
knowledge_bases = {}

def categorize_pdfs(pdf_list):
    category_map = {}
    for pdf in pdf_list:
        prefix = os.path.basename(pdf).split('_')[0]  # Extract prefix from the filename, not the path

        # Mapping prefixes to categories
        if prefix == "Auto":
            category = "Autoverzekering"
        elif prefix == "AVB":
            category = "Bedrijfsaansprakelijkheidsverzekering"
        elif prefix == "BestAVB":
            category = "AVB Bestuurders"
        elif prefix == "BS":
            category = "Bedrijfsschadeverzekering"
        elif prefix == "BestAuto":
            category = "Bestelautoverzekering"
        elif prefix == "Brand":
            category = "Brandverzekeringen"
        elif prefix == "Cara":
            category = "Caravanverzekering"
        elif prefix == "EigVerv":
            category = "Eigen vervoer"
        elif prefix == "Fiets":
            category = "Fietsverzekering"
        elif prefix == "Geb":
            category = "Gebouwen"
        elif prefix == "GW":
            category = "Goed Werkgeverschap"
        elif prefix == "IB":
            category = "Inboedelverzekering"
        elif prefix == "Inv":
            category = "Inventaris"
        elif prefix == "Mot":
            category = "Motorverzekering"
        elif prefix == "RB":
            category = "Rechtsbijstandverzekering"
        elif prefix == "Reis":
            category = "Reisverzekering"
        elif prefix == "Scoot":
            category = "Scootmobielverzekering"
        elif prefix == "WEGAS":
            category = "WEGAS"
        elif prefix == "WerkMat":
            category = "Werk- en landbouwmaterieelverzekering"
        elif prefix == "WEGAM":
            category = "Werkgeversaansprakelijkheid Motorrijtuigen (WEGAM)"
        elif prefix == "Woon":
            category = "Woonhuisverzekering"
        else:
            category = "others"

        # Add the pdf to its category in the map
        if category not in category_map:
            category_map[category] = []
        category_map[category].append(pdf)

    return category_map

def main():
    st.header("VA-Polisvoorwaardentool")

    # Get the API key from Streamlit's secrets
    api_key = st.secrets["OPENAI_API_KEY"]

    # Set it as an environment variable (if required by any library)
    os.environ["OPENAI_API_KEY"] = api_key

    # Get list of preloaded PDFs recursively
    pdf_dir = "preloaded_pdfs/"
    all_pdfs = [os.path.join(dp, f) for dp, dn, filenames in os.walk(pdf_dir) for f in filenames if f.endswith('.pdf')]

    category_map = categorize_pdfs(all_pdfs)

    categories = list(category_map.keys())
    if not categories:
        st.warning("Geen polisvoorwaarden gevonden in deze categorie.")
        return

    # Get list of categories and let the user choose
    selected_category = st.selectbox("Kies een categorie:", categories)

    # Get list of PDFs for the selected category
    available_pdfs = category_map[selected_category]
    pdf_names = [os.path.basename(pdf) for pdf in available_pdfs]  # Extract the names for display
    selected_pdf_name = st.selectbox("Welke polisvoorwaarden wil je raadplegen?", pdf_names)

    # Map the selected name back to its path
    selected_pdf_path = available_pdfs[pdf_names.index(selected_pdf_name)]

      # Display the selected PDF
    if selected_pdf_path:
        display_pdf(selected_pdf_path)
    # Check if embeddings for this PDF are already cached
    if selected_pdf_path not in knowledge_bases:
        # Read the selected PDF
        with open(selected_pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            # Split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            # Create embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_bases[selected_pdf_path] = FAISS.from_texts(chunks, embeddings)
    # Use the cached/embedded knowledge base for the selected PDF
    knowledge_base = knowledge_bases[selected_pdf_path]
    # Show user input
    user_question = st.text_input("Stel een vraag over de polisvoorwaarden")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            # You can print other debugging info if needed, but avoid printing the API key.
            print(cb)
        st.write(response)

if __name__ == '__main__':
    main()
