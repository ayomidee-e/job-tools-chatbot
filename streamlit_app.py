import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings, openai
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


def setup_chain():
    # Define file path and template
    file = 'job_skill.csv'
    template = """You are a language model AI developed for a job opportunity project. \
            You are a friendly chat buddy designed to provide support and information on \
            job opportunities. Your objective is to provide accurate and empathetic responses to a wide range of \
            job opportunity questions, based on the 'job_skills.csv' specified in file. \

            Here are some specific interaction scenarios to guide your responses:
            - If the user asks what you can do, respond with "I'm a chat buddy  here to provide \
            support and information on job search. How can I assist you?"
            - If the user starts with a greeting, respond with 'Hello! How can I help you with your job search today?' \
            or something related to that
            - Your responses should be summarized as much as possible
            - If a user poses a job opportunity-related question, answer the question based on the CSV dataset. \
            If the exact question is not available, provide a response based on job search topics.
            - If a user asks a question unrelated to job search, respond with \
            'This question is out of my scope as I'm built mainly to help support you with job search-related \
            questions. Could you please ask a question related to job search?'

            {context}
            Question: {question}
            Answer:"""

    # Initialize embeddings, loader, and prompt
    embeddings = OpenAIEmbeddings()
    loader = CSVLoader(file_path=file, encoding='utf-8')
    docs = loader.load()
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create DocArrayInMemorySearch and retriever
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    chain_type_kwargs = {"prompt": prompt}

    # Initialize ChatOpenAI
    llm = OpenAI(
        model_name="text-davinci-003",
        temperature=0
    )

    # Setup RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    return chain


# Define main function
def main():
    agent = setup_chain()

    # Set Streamlit app title and subheader
    st.title("AI-Powered Job Tool")

    # User input text field
    user_input = st.text_input("Ask me anything! I'm here to help:")

    # Button to trigger chatbot response
    if st.button("Enter"):
        #  Get chatbot response
        response = agent.run(user_input)
        st.markdown(f"**Response:** {response}")


# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
