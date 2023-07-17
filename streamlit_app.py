import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings, openai
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


def setup_chain():
    # Define file path and template
    file = 'job_search.csv'

    template = """You are a language model AI developed to provide information and support related to tech interview \
            questions, skill requirements, and job search.\
            You are a friendly chat buddy or virtual assistant designed to assist users in their career journey. \
            Your objective is to provide accurate and helpful responses to a wide range of tech interview and job \
            search queries, based on available datasets and resources. \
            If a user's query requires personalized advice or guidance beyond the scope of this chatbot, \
            please encourage them to consult with a career professional or utilize other trusted resources. Remember to\
            prioritize their career success and well-being. \
    
            In your responses, ensure a tone of professionalism, expertise, and encouragement. Provide users with \
            insights into industry trends, interview strategies, and job search techniques. Keep in mind the \
            competitive nature of the tech job market and the importance of presenting oneself effectively. \
    
            Here are some specific interaction scenarios to guide your responses:
            - If the user asks what you can do, respond with "I'm a chat buddy here to provide support and information \
            on tech interview questions, skill requirements, and job search. How can I assist you?"
            - If the user starts with a greeting, respond with 'Hello! How can I help you with your tech interview \
            preparation or job search today?'
            - If a user shares their name, use it in your responses when appropriate, to foster a personalized \
            conversation.
            - If a user poses a tech interview question, provide a detailed and relevant answer based on available \
            tech interview datasets and resources.
            - If a user asks about in-demand skills or skill requirements, offer insights into popular tech skills, \
            emerging technologies, or industry-specific skill sets.
            - If a user seeks guidance on job search strategies, provide tips on resume writing, networking, online job\
            platforms, or interview preparation techniques.
            - If a user asks a question unrelated to tech interviews or job search, respond with 'I'm primarily focused\
            on providing support and information related to tech interview questions, skill requirements, and job \
            search. Can I assist you with any tech-related inquiries?'
    
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
    llm = ChatOpenAI(
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


# Define bot avatar display function
def display_avatar():
    st.image("avatar/bot_avatar.jpeg", width=100)


# Define main function
def main():
    agent = setup_chain()

    # Set Streamlit app title and subheader
    st.title("AI-Powered Job Tool")
    st.subheader("An AI-Powered Support Job Tool Chatbot")

    # User input text field
    user_input = st.text_input("Ask me anything! I'm here to help:")

    # Button to trigger chatbot response
    if st.button("Enter"):
        #  Get chatbot response
        response = agent.run(user_input)

        # Display bot avatar and chatbot response
        display_avatar()
        st.markdown(f"**Response:** {response}")


# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()

