import os
import time
import hashlib
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

if not os.environ["GOOGLE_API_KEY"]:
    st.error("API KEY IS NOT ENTERED")
    st.stop()
    
MODEL = "gemini-2.5-flash"
EMB_MODEL = "gemini-embedding-001"
UPDATE_INTERVAL = 3600

urls = [
    "https://www.myscheme.gov.in/",
    "https://www.startupindia.gov.in/content/sih/en/government-schemes.html",
    "https://msme.gov.in/schemes",
    "https://dst.gov.in/call-for-proposals"
]

st.title("Government Grant RAG Assistant 🤖")


def scrape_websites():
    docs = []

    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load {url}")

    return docs


def get_hash(docs):
    text = "\n".join(d.page_content for d in docs)

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMB_MODEL
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local("faiss_index")

    return vectorstore


@st.cache_resource
def initialize():

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMB_MODEL
    )

    if os.path.exists("faiss_index/index.faiss"):

        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    docs = scrape_websites()
    current_hash = get_hash(docs)

    with open("website_hash.txt", "w") as f:
        f.write(current_hash)

    with open("last_update.txt", "w") as f:
        f.write(str(time.time()))

    return create_vectorstore(docs)


def check_for_updates(vectorstore):

    if not os.path.exists("last_update.txt"):
        return vectorstore

    with open("last_update.txt", "r") as f:
        last_update = float(f.read())

    if time.time() - last_update < UPDATE_INTERVAL:
        return vectorstore

    st.info("Checking government websites for updates...")

    docs = scrape_websites()
    new_hash = get_hash(docs)

    old_hash = ""

    if os.path.exists("website_hash.txt"):
        with open("website_hash.txt", "r") as f:
            old_hash = f.read().strip()

    if new_hash != old_hash:

        st.success("New website data detected. Updating chatbot...")

        vectorstore = create_vectorstore(docs)

        with open("website_hash.txt", "w") as f:
            f.write(new_hash)

    else:
        st.info("No new updates found.")

    with open("last_update.txt", "w") as f:
        f.write(str(time.time()))

    return vectorstore


vectorstore = initialize()

vectorstore = check_for_updates(vectorstore)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a Government Grant and Scheme Assistant.

Answer ONLY using the provided government website context.

Do not invent any information.

If the answer is not available in the context, say:

"I could not find enough information in the available government websites."

Keep different schemes separate.

For each scheme provide:

Scheme Name:
Department / Ministry:
Purpose:
Eligibility:
Financial Support:
Application Process:
Important Details:
Source Website:
"""
    ),
    (
        "human",
        """
Context:

{context}

Question:

{question}
"""
    )
])


def format_docs(docs):
    return "\n\n-------------------------\n\n".join(
        d.page_content for d in docs
    )


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


question = st.text_input(
    "Enter Your Query Regarding Government Schemes"
)

search_button = st.button(
    "Search Government Schemes 🔍"
)

if search_button:

    if question.strip():

        with st.spinner("Searching Government Schemes..."):

            response = rag_chain.invoke(question)

        st.subheader("Government Scheme Results")

        st.write(response)

    else:

        st.error("ENTER QUESTION")
