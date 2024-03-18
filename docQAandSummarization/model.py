# Import necessary packages
import os
#from langchain_text_splitters import TextSplitter
from langchain.text_splitter import TextSplitter
import openai
from langchain_community.document_loaders import PyPDFLoader
#from pypdf import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import Pinecone, PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from pinecone import Pinecone,PodSpec
from langchain_pinecone import PineconeVectorStore

def initialize_openai_api(api_key, base_url):
    openai.api_key = api_key
    openai.base_url = base_url

def upload_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    print("Loading pdf")
    pages = loader.load_and_split()
    return pages

# def split_documents(pages):
    
#     text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_documents(pages)
#     return chunks
def split_text_into_chunks(texts):
    print("making chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(texts)
    return chunks

def initialize_embeddings_model(model_name):
    print("Embedding...")
    embeddings_model = SentenceTransformerEmbeddings(model_name=model_name)
    return embeddings_model

import pinecone

# Initialize Pinecone with API key
# pinecone_api_key = "1c6bf2f6-c246-4ea6-b1e5-93a371713524"
# pinecone.init(api_key=pinecone_api_key)
# pinecone_instance = pinecone.Pinecone(api_key=pinecone_api_key)

# Create a Pinecone vector store without passing the API key
#pvs = PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)


def initialize_pinecone_index(api_key, index_name, dimension=384, metric='cosine', environment='gcp-starter'):
    # pinecone = Pinecone()
    # try:
    #     pinecone.create_index(name=index_name, dimension=dimension, metric=metric, environment=environment)
    # except:
    #     print("INDEX CREATION FAILED")
    # pinecone = Pinecone(api_key = '1c6bf2f6-c246-4ea6-b1e5-93a371713524')
    pinecone = Pinecone(api_key = api_key)
    index_name = index_name
    print("initializing_pinecone_index")
    if index_name in [index.name for index in pinecone.list_indexes()]:
        pinecone.delete_index(index_name)
    try:
            pinecone.create_index(name = index_name,
                                dimension = 384 ,
                                metric = metric,
                                spec = PodSpec(environment='gcp-starter'))
    except:
            print("INDEX CREATION FAILED")

def store_text_embeddings_in_pinecone(texts, embeddings_model, index_name,pinecone_api_key):
    print("store_text_embeddings_in_pinecone")
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    pvs = PineconeVectorStore.from_documents(texts, embeddings_model, index_name=index_name)
    
    return pvs

def initialize_chat_model(openai_api_key, base_url, model_name, temperature=0, max_tokens=1000):
    print("initializibg_chat_model")
    llm = ChatOpenAI(openai_api_key=openai_api_key, base_url=base_url, model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    
    return llm

def initialize_conversation_memory(memory_key="chat history"):
    memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)
    print("intializing memory")
    return memory

def initialize_retrieval_qa_chain(llm, retriever, memory, prompt_template):
    print("Initializing retreival qa chain")
    qa_chain_prompt = PromptTemplate.from_template(prompt_template)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, memory=memory, chain_type_kwargs={"prompt": qa_chain_prompt})
    
    return qa_chain

def ask_question(question, qa_chain):
     result = qa_chain({"query": question})
     
     return result["result"]

def initialize_conversational_retrieval_chain(llm, retriever, memory):
    print("initializing_conversational_retrieval_chain")

    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    
    return qa

def summarize_text(llm, texts):
    print("Summarizing")
    prompt_template = """write a detailed summary with the following text,Atleast 30 lines of summary and key points,
    {texts}  Detailed summary"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.run(texts)
    
    return summary
def split_documents(pages):
    
    print("Splitting documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    
    return chunks

# Set API keys and parameters
openai_api_key = 'LL-c8QuMCAefOGrl3pNb0GjfTbNm8Oz09YDW2AiI5RuMmldVfERMnxZJKB0saBzBUc7'
openai_base_url = "https://api.llama-api.com"
pinecone_api_key = '1c6bf2f6-c246-4ea6-b1e5-93a371713524'
pinecone_index_name = "praveen"
pdf_path = 'LLaMA2_Paper.pdf'
model_name = "all-MiniLM-L6-v2"


def intialize():

     
    initialize_openai_api(openai_api_key, openai_base_url)

    embeddings_model = initialize_embeddings_model(model_name)
    initialize_pinecone_index(pinecone_api_key, pinecone_index_name)

    llm = initialize_chat_model(openai_api_key, openai_base_url, "mistral-7b-instruct", temperature=0, max_tokens=3000)

    memory=initialize_conversation_memory()

    return embeddings_model, llm, memory

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer . Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""
#memory = initialize_con
#print(pages)





# summary=summarize_text(llm,texts[0:5])
# print(summary)

# qa_model=initialize_retrieval_qa_chain(llm, pvs.as_retriever(), memory, prompt_template)
# print(ask_question(qa_model))
