import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
import io

# Initialize the OpenAI LLM with your API key
openai_api_key = st.secrets["OPENAI_API_KEY"]
model = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4-turbo-preview", temperature=0.20)

# Load the QA chain with the map_reduce type
qa_chain = load_qa_chain(llm, chain_type="map_reduce")

# Setup the AnalyzeDocumentChain with the QA chain
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

def main():
    st.title("Polisvoorwaardentool")

    # Allow users to upload a PDF document
    uploaded_file = st.file_uploader("Choose a document", type=["pdf"])
    question = st.text_input("Enter your question here")

    if uploaded_file is not None and question:
        # Extract text from the uploaded PDF document
        input_document = extract_text_from_pdf(uploaded_file)

        # Run the QA document chain on the uploaded document and the user's question
        answer = qa_document_chain.run(
            input_document=input_document,
            question=question,
        )

        # Display the answer to the user
        st.write("Answer:", answer)

def extract_text_from_pdf(uploaded_file):
    """Extract text from the uploaded PDF file."""
    document_text = ""
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    for page in reader.pages:
        text = page.extract_text()
        if text:
            document_text += text + "\n"
    return document_text

if __name__ == "__main__":
    main()
