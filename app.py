import streamlit as st
from PyPDF2 import PdfReader
# Ensure these imports match the installed langchain packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = "preloaded_pdfs/PolisvoorwaardenVA"

# Assuming openai_api_key is stored securely in Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize ChatOpenAI model
model = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4-turbo-preview", temperature=0.20)

def extract_text_from_pdf(file_path):
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
    return document_text

def generate_summary(document_text, user_question):
    # Craft the prompt using ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_template(f"{{document_text}}\n\n### Vraag:\n{user_question}\n### Antwoord:")
    full_prompt = prompt_template.format(document_text=document_text)
    
    # Generate response
    response = model.generate([full_prompt], max_tokens=500)  # Adjust max_tokens if needed
    output_parser = StrOutputParser()
    summary = output_parser.parse(response)
    return summary

def main():
    st.title("Polisvoorwaardentool")
    
    # Your existing logic for category and document selection
    
    user_question = st.text_input("Vraag:")
    
    if user_question:
        # Assuming document_path is defined based on user selection
        document_text = extract_text_from_pdf(document_path)
        summary = generate_summary(document_text, user_question)
        st.text_area("Antwoord", summary, height=300)

if __name__ == "__main__":
    main()
