from langchain_community.vectorstores import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata


class ChatPDF:
    vector_store = None
    retriever = None
    chain  = None

    def __init__(self):
        self.model = ChatOllama(model="llama3")  # use the latest llama3 model

        # split the text into chunks using the given chunk size and chunk overlap.
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100, add_start_index=True)

        # Create the prompt template
        self.prompt = PromptTemplate.from_template(
             """
            [INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
            Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
            )

    def ingest(self, pdf_path):
        # load the PDF file
        docs = PyPDFLoader(file_path=pdf_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks) # Filter out metadata types that are not supported for a vector store.

        # Store the embeddings in a vector DB - Chroma in this case
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings()
            )

        # set a retriever for a vector store to perform similarity based searches
        self.retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    'k': 3,
                    'score_threshold': 0.5
                },
            )

        self.chain = ({
                "context": self.retriever,
                "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
            )

    def ask(self,query:str):
        if not self.chain:
            return "Please ingest a PDF file first."
        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None


