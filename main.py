import google.generativeai as genai
import os
import streamlit as st
import pickle
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

api_key = "YOUR_API_KEY_HERE"
genai.configure(api_key=api_key)

st.title("Article Research Tool 📊")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
index_folder = "faiss_index_folder"

main_placeholder = st.empty()
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.9, max_tokens=500, google_api_key=api_key)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorindex_genai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Building...✅✅✅")
    time.sleep(2)

    vectorindex_genai.save_local(index_folder)
    main_placeholder.text("Index Saved Locally at 'faiss_index_folder'")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(index_folder):
        loaded_vectorindex_genai = FAISS.load_local(
            index_folder,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=loaded_vectorindex_genai.as_retriever())

        result = chain({"question": query}, return_only_outputs=True)

        
        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  
            for source in sources_list:
                st.write(source)
