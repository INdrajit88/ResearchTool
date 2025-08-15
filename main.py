import os
import asyncio
import streamlit as st
import time
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader

# Set up event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["api_key"] = st.secrets['GOOGLE_API_KEY']

st.title("Research Tool")
st.sidebar.title("New Article Url")

# Create a directory for storing the index
index_path = "faiss_store"
os.makedirs(index_path, exist_ok=True)

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"Enter url {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading data...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
    )
    main_placeholder.text("splitting docs...")
    docs = text_splitter.split_documents(data)

    main_placeholder.text("Creating Embeddings...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Started....")
    time.sleep(2)

    # Save using FAISS native method
    vectorstore.save_local(index_path)

query = main_placeholder.text_input("Enter question")
if query:
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        # Changed 'query' to 'question' in the input dictionary
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])
        #display source
        sources=result.get("sources", "")
        if sources:
            st.subheader("sources")
            sources_list=sources.split("\n")
            for source in sources_list:
                st.write(source)
