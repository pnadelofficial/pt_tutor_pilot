import streamlit as st
import pandas as pd
import re
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
import os
from prompts import *
import random

st.title('Tufts Physical Therapy AI Tutor')

@st.cache_resource
def prep_embeddings():
    hf= HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en",
        encode_kwargs={'normalize_embeddings': True}
    )
    return hf

@st.cache_resource
def prep_model(choice, _qa_prompt, _doc_prompt, _db):
    if choice.startswith('GPT'):
        llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k-0613', openai_api_key=oak) 
        qa_chain = load_qa_with_sources_chain(
            llm, chain_type='stuff',
            prompt=_qa_prompt,
            document_prompt=_doc_prompt
        )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key='question', output_key='answer')
        qa = RetrievalQAWithSourcesChain(
            combine_documents_chain=qa_chain, retriever=_db.as_retriever(),
            reduce_k_below_max_tokens=True, return_source_documents=True,
            memory=memory
        )
    else:
        hat = st.text_input('Input HuggingFace API Key')
        if hat != '':
            repo_id = "tiiuae/falcon-7b-instruct"
            llm = HuggingFaceHub(huggingfacehub_api_token=hat, 
                                repo_id=repo_id, 
                                model_kwargs={"temperature":0.1})
            qa_chain = load_qa_with_sources_chain(
                llm, chain_type='stuff',
                document_prompt=_doc_prompt
            )
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key='question', output_key='answer')
            qa = RetrievalQAWithSourcesChain(
                combine_documents_chain=qa_chain, retriever=_db.as_retriever(),
                reduce_k_below_max_tokens=True, return_source_documents=True,
                memory=memory
            )
    return qa, memory

hf = prep_embeddings()
model_choice = st.selectbox("What model would you like to use?", ["GPT3.5", "Falcon 7B", "Claude2 (TBD)"])
doc_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"]
)

def format_names(name):
    if name == 'pt_example_db':
        return 'PT Example DB'
    else:
        return name.replace('_', ' ').title().replace('Db', 'DB')

db_choice = st.selectbox("What week's material would you like to search?", os.listdir('./dbs'), format_func=format_names)
db = FAISS.load_local(f'dbs/{db_choice}', hf)

template_radio = st.radio(
    "What systemp rompt would you like to use",
    ["Default LangChain prompt", "Use my own"]
)

if template_radio == "Default LangChain prompt":
    template = st.text_area("System Prompt", LANGCHAIN_DEFAULT_QA)
    qa_prompt = PromptTemplate.from_template(template)
else:
    template = st.text_area("System Prompt")
    qa_prompt = PromptTemplate.from_template(template)
    
with st.sidebar:
    
    oak = st.text_input('Input OpenAI API Key') 

if oak != '':
    st.success('Key entered!')
    qa, memory = prep_model(model_choice, qa_prompt, doc_prompt, db)

def clean_cite(text):
    text = re.sub('\d+\:\d+\:\d+','',text)
    text = re.sub('\[.*?\]','',text)
    text = text.replace('cielo24 | what’s in your video? | cielo24.com', '')
    return text

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if question := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            result = qa({"question":question, "chat_history":memory.chat_memory})
        response = result['answer']
        cites = result['source_documents']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(response, unsafe_allow_html=True)

        with col2: 
            with st.expander("**Citations**"):
                cite_df = pd.DataFrame(cites)
                cites_meta = list(zip(list(c[1] for c in cite_df.drop_duplicates(subset=0)[0]), list(c[1] for c in cite_df.drop_duplicates(subset=0)[1])))
                for cite in cites_meta:
                    st.markdown(f"*{cite[1]['source'].split('.pdf')[0]}*")
                    st.markdown(clean_cite(cite[0]), unsafe_allow_html=True)             
    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button('Reset chat'):
    st.session_state.messages = []
    memory = ConversationBufferMemory()
    st.experimental_rerun()

with st.sidebar:
    st.divider()
    st.write('\nRemember to give us feedback!\n')
    with st.form("student_feedback"):
        st.write("Feedback")
        likert1 = st.slider("Ease of use", min_value=0, max_value=5, value=3)
        likert2 = st.slider("Accuracy", min_value=0, max_value=5, value=3)
        likert3 = st.slider("Overall", min_value=0, max_value=5, value=3)
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            os.makedirs('./feedback', exist_ok=True)
            id = f"{len(os.listdir('./feedback'))}_{random.randint(10_000, 20_000)}"
            with open(f"./feedback/{id}_feedback.csv", "w") as f:
                f.write(f"ID, Ease of use, Accuracy, Overall\n")
                f.write(f"{id}, {likert1}, {likert2}, {likert3}\n")
            
        if os.path.exists(f"./feedback/{id}_feedback.csv"):
            df = pd.read_csv(f"./feedback/{id}_feedback.csv")
            st.dataframe(df)