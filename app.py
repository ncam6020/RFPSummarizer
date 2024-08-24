from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Sidebar contents
with st.sidebar:
    st.title('💬PDF Summarizer and Q/A App')
    st.markdown('''
    ## About this application
    You can built your own customized LLM-powered chatbot using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(2)
    st.write('Why drown in papers when your chat buddy can give you the highlights and summary? Happy Reading.')
    add_vertical_space(2)    
    st.write('Made by ***Sangita Pokhrel***')

def main():
    load_dotenv()

    # Main Content
    st.header("Ask About Your PDF 🤷‍♀️💬")

    # Upload file
    pdf = st.file_uploader("Upload your PDF File and Ask Questions", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,  # Reduced chunk size
            chunk_overlap=100,  # Adjust overlap
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 0]  # Filter out empty chunks
        
        # Create embeddings in batches
        batch_size = 10
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            try:
                embeddings = OpenAIEmbeddings().embed_documents(batch)
                all_embeddings.extend(embeddings)
            except ValueError as e:
                st.error(f"An error occurred: {str(e)}")
                break
        
        if all_embeddings:
            knowledge_base = FAISS.from_embeddings(all_embeddings)
        
            # Show user input
            user_question = st.text_input("Please ask a question about your PDF here:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)
                
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    print(cb)
                
                st.write(response)

if __name__ == '__main__':
    main()
