__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import re
import pypdf as pdf
from langchain.vectorstores import Chroma
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO

st.title('Tufts Physical Therapy AI Tutor')

@st.cache_resource
def prep_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )
    return hf

@st.cache_resource(experimental_allow_widgets=True)
def prep_model(choice, _qa_prompt, _doc_prompt, _db):
    if choice.startswith('GPT'):
        oak = st.text_input('Input OpenAI API Key')
        if oak != '':
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
            return qa, memory      
    else:
        hat = st.text_input('Input HuggingFace API Key')
        if hat != '':
            repo_id = "tiiuae/falcon-7b-instruct"
            llm = HuggingFaceHub(huggingfacehub_api_token=hat, 
                                repo_id=repo_id, 
                                model_kwargs={"temperature":0.1})
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
            return qa, memory

hf = prep_embeddings()
uploaded_files = st.file_uploader("Choose some PDFs to upload", accept_multiple_files=True)
print(bool(uploaded_files))
if uploaded_files:
    docs = []
    metadatas = []
    for uploaded_file in uploaded_files:
        read_pdf = pdf.PdfReader(BytesIO(uploaded_file.getvalue()))

        texts = ""
        for page in read_pdf.pages:
            texts += page.extract_text()
        docs.append(texts)
        metadatas.append(uploaded_file.name)
    df = pd.DataFrame({"source":metadatas, "text":docs})
    loader = DataFrameLoader(df, page_content_column='text')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_documents(documents)
    for i, text in enumerate(texts):
        if "source" not in text.metadata:
            text.metadata["source"] = f"{i}-pl"

choice = st.selectbox("What model would you like to use?", ["GPT3.5", "Falcon 7B", "Claude2 (TBD)"])
template = st.text_area("System Prompt",
"""
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
Source: 28-pl
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
Source: 30-pl
Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
Source: 4-pl
=========
FINAL ANSWER: This Agreement is governed by English law.
SOURCES: 28-pl

QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
Source: 0-pl
Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
Source: 24-pl
Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
Source: 5-pl
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
Source: 34-pl
=========
FINAL ANSWER: The president did not mention Michael Jackson.
SOURCES:

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:
"""
)

if template != '':
    qa_prompt = PromptTemplate.from_template(template)
doc_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"]
)

if (template != ''):
    if uploaded_files:
        db = Chroma.from_documents(texts, hf)        
    else:
        db = Chroma(persist_directory='./p_db', embedding_function=hf)
    qa, memory = prep_model(choice, qa_prompt, doc_prompt, db)

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
            st.markdown(f"*{cite[1]['source'].split('.pdf')[0]}*") #, page {cite[1]['page']+1}
            st.markdown(clean_cite(cite[0]), unsafe_allow_html=True)             
    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button('Reset chat'):
    st.session_state.messages = []
    memory = ConversationBufferMemory()
    st.experimental_rerun()
