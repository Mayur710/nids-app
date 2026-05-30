from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#loading all txt files from security docs using directory loader
loader = DirectoryLoader('security_docs',
                        glob='**/*.txt',
                        show_progress=True, 
                        loader_cls=TextLoader,
                        loader_kwargs={'encoding': 'utf-8'})
#splitting the documents into smaller chunks using recursive character text splitter with chunk size of 500
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#loading the documents and splitting them into chunks
documents = loader.load()
split_docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory='chroma_db')