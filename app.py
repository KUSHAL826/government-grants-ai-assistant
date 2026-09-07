import os
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
import streamlit as st
os.environ["GOOGLE_API_KEY"]=os.getenv("GEMINI_API_KEYS")
if not os.environ["GOOGLE_API_KEY"]:
    st.error("API KEY IS NOT ENTERED")
    st.stop()
MODEL="gemini-3.5-flash-lite"
EMB_MODEL="gemini-embedding-001"


st.title("Government Grant RAG Assistant 🤖")
question=st.text_input("Enter Your Query Regarding Government Schemes")
search_button=st.button("Search Government Schemes 🔍")
#1.1Configuring PDF
pdf_name="government_grants_testing.pdf"
#1.2Extracting text from pdf
docs=PyPDFLoader(pdf_name).load()
print("NUMBER OF PAGES: ",len(docs))
#1.3chunking
splitters=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)
chunks=splitters.split_documents(docs)
print(f"CHUNKING DONE WITH NUMBER OF CHUNKS:{len(chunks)}")

#1.4Embeddings vectors faiss 
embeddings=GoogleGenerativeAIEmbeddings(model=EMB_MODEL)
vectorstore=FAISS.from_documents(chunks,embeddings)

#1.4 Retrieving
retriever=vectorstore.as_retriever(search_kwargs={'k':3})


#2 From retrieved chunks collecting answer and formatting as per user input
llm=ChatGoogleGenerativeAI(model=MODEL)
prompt=ChatPromptTemplate([
    ('system',"""
You are a PDF assistant.

Answer the user's question using ONLY the information provided
in the document context.

Do not invent or add information that is not present in the context.

If the context does not contain enough information, say:
"I could not find enough information in the document."

return clean formatted text so schemes with its details will not get mixed
"""),
    ('human','context from the document:{context} ----question:{question}')
])
def format_docs(docs):
    return ("\n\n-----------------\n\n".join(d.page_content for d in docs))

rag_chain=(
    {
        'context':retriever | format_docs,
        'question':RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
if search_button:
    if question.strip():
        with st.spinner("Searching for Government Schemes"):
            response=rag_chain.invoke(question)

        st.subheader("FETCHING........")
        st.write(response)
    else:
        st.error("ENTER QUESTION")
