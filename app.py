import streamlit as st
import pandas as pd
import re
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

st.title('Tufts Physical Therapy AI Tutor')

@st.cache_resource
def prep_model():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_kwargs = {'device': 'mps'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        # model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = Chroma(persist_directory='./p_db', embedding_function=hf)
    huggingfacehub_api_token = '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'
    repo_id = "tiiuae/falcon-7b-instruct"
    llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                        repo_id=repo_id, 
                        model_kwargs={"temperature":0.1, "max_new_tokens":2000})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key='question', output_key='answer')
    qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(), return_source_documents=True, memory=memory)
    return qa, memory, hf
qa, memory, hf = prep_model()

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
        st.markdown(response, unsafe_allow_html=True)
        cite_df = pd.DataFrame(cites)
        cites_meta = list(zip(list(c[1] for c in cite_df.drop_duplicates(subset=0)[0]), list(c[1] for c in cite_df.drop_duplicates(subset=0)[1])))
        st.markdown("**Citations**")
        for cite in cites_meta:
            st.markdown(f"*{cite[1]['source'].split('.pdf')[0]}, page {cite[1]['page']+1}*")
            st.markdown(clean_cite(cite[0]), unsafe_allow_html=True)             
    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button('Reset chat'):
    st.session_state.messages = []
    memory = ConversationBufferMemory()
    st.experimental_rerun()
