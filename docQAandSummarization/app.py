import streamlit as st
st.set_page_config(page_title="SummarEase", layout = "wide")#, #page_icon="download.png")
import base64
from streamlit_lottie import st_lottie
from st_copy_to_clipboard import st_copy_to_clipboard
# from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
# st.balloons()
# st.markdown(
#     """
#     <style>
#     body{
#     background-color:lightblue;
#     }
#     </style>"""
# , unsafe_allow_html = True)
# from langchain.prompts import PromptTemplate
# # from langchain_text_splitters import TextSplitter
# from langchain_pinecone import Pinecone, PineconeVectorStore
# from model import initialize_retrieval_qa_chain, ask_question, initialize_conversation_memory, summarize_text, upload_pdf

import model
import random
from model \
    import *


def spinn_content():
    spin_text = random.choice([
    "Loading humor...",
    "I'm bored",
    "Procrastinating...",
    "Counting unicorns...",
    "Reticulating splines...",
    "Convincing the bits to line up...",
    "Finding the lost socks...",
    "Brewing coffee...",
    "Gathering magic...",
    "Training ninja squirrels...",
    "Assembling Avengers...",
    "Summoning dragons...",
    "Unraveling mysteries...",
    "Locating Waldo...",
    "Hiding from the compiler...",
    "Polishing the pixels...",
    "Preparing the popcorn...",
    "Unjamming the printer...",
    "Chasing bugs...",
    "Wrangling cats...",
    "Consulting the magic 8-ball...",
    "Thinking up witty banter..."
        ]
        )
    return spin_text

with st.sidebar:
    st.title(' LLM QA & SUMMARIZATION App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/) UI
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models,https://www.llama-api.com/) LLM model
    - [pinecone](https://app.pinecone.io/) Vector database
 
    ''')
    add_vertical_space(5)
    st.write('Powered by Indium Software')
 
# load_dotenv()
    
temp_dir = r"D:\Data\Official\PDf_qa\Uploaded_docs"

def save_file(uploaded_file):
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    print("FILE SAVED HERE", path)

    return path
 
st.title("Question Answering and Summarization App")
st.write("Unlock the power of summarization and Q&A with our intuitive app! Say goodbye to lengthy texts and hello to concise summaries. Whether it is articles, reports, or documents, our app simplifies complex information, saving you time and effort. Experience the convenience of instant insights and make informed decisions effortlessly. Try it now and streamline your reading experience!")

file = st.file_uploader("Drop your file...")

if 'setup' not in st.session_state:
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_awP420Zf8l.json"
    # st_lottie(lottie_url, key="user")
    st_lottie(lottie_url,
    reverse=True,
    height=400,
    width=1000,
    speed=0.3,
    loop=True,
    quality='high',
    key='img')
  
    
    
    
    # if 'spin_text' not in st.session_state:
    #     st.session_state['spin_text'] = spin_text
    # else:
    #     spin_text = st.session_state['spin_text']


    with  st.spinner(spinn_content()):
        embeddings_model, llm, memory = intialize()
        

    
        
        pdf_path = save_file(file)
    
    with  st.spinner(spinn_content()):
        pages = upload_pdf(pdf_path)

    with  st.spinner(spinn_content()):
        texts = split_documents(pages)
    
    with  st.spinner(spinn_content()):
        chunks = split_text_into_chunks(texts)
    
    with  st.spinner(spinn_content()):
        pvs = store_text_embeddings_in_pinecone( chunks, embeddings_model, pinecone_index_name, pinecone_api_key = pinecone_api_key)
    
    with  st.spinner(spinn_content()):    
        qa_model=initialize_retrieval_qa_chain(llm, pvs.as_retriever(), memory, prompt_template)
        st.session_state['setup'] = True
        st.session_state['values'] = qa_model
        st.session_state['llm'] = llm
        st.session_state['texts'] = texts
# pdf_path,pages,texts,chunks,pvs,llm,
        st.write("OKAY I'M READY")


# Uploading pdf file

st.divider()
col1, col2 = st.columns(2)
with col1:   
    st.header("Question And Answer")   
    name = st.text_input("Enter Your Question : ")
    # if 'question' not in st.session_state:
    #     st.session_state['question']=name

    # question = st.session_state['question']


    if(st.button('Enter')):
        if 'values' in st.session_state:
            qa_model = st.session_state['values']
        answer = ask_question(name, qa_model)
        result = answer
        st.success(result)
        st_copy_to_clipboard(result)

with col2:
    st.header("Summary") 
    add_vertical_space(2)
    if(st.button('Summary')):
        llm = st.session_state['llm']
        texts = st.session_state['texts']
        summary=summarize_text(llm,texts[0:5])
        st.success(summary)
        st_copy_to_clipboard(summary)

        st.success("success!!!!👍") 
)
