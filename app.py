import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def categorize_pdfs(pdf_list):
    category_map = {}
    for pdf in pdf_list:
        prefix = pdf.split('/')[-1].split('_')[0]  # Extract prefix from the filename, not the path
        
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
    # Get the API key from Streamlit's secrets
    api_key = st.secrets["OPENAI_API_KEY"]
    
    # Set it as an environment variable (if required by any library)
    os.environ["OPENAI_API_KEY"] = api_key

    st.set_page_config(page_title="VA-Polisvoorwaardentool")
    st.header("VA-Polisvoorwaardentool")

    # Get list of preloaded PDFs recursively
    pdf_dir = "preloaded_pdfs/"
    all_pdfs = [os.path.join(dp, f) for dp, dn, filenames in os.walk(pdf_dir) for f in filenames if f.endswith('.pdf')]
    category_map = categorize_pdfs(all_pdfs)

    categories = list(category_map.keys())
    if not categories:
        st.warning("No PDFs found in the expected categories.")
        return

    # Get list of categories and let the user choose
    selected_category = st.selectbox("Choose a category:", categories)

    # Get list of PDFs for the selected category
    available_pdfs = category_map[selected_category]
    pdf_names = [os.path.basename(pdf) for pdf in available_pdfs]  # Extract the names for display
    selected_pdf_name = st.selectbox("Welke polisvoorwaarden wil je raadplegen?", pdf_names)

    # Map the selected name back to its path
    selected_pdf_path = available_pdfs[pdf_names.index(selected_pdf_name)]

    # Read the selected PDF
    with open(selected_pdf_path, "rb") as f:  # Use the full path here
        pdf_reader = PdfReader(f)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into chunks
        text_spliter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_spliter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

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
