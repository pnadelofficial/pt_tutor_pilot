import streamlit as st
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import os
import re

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
        llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k-0613', openai_api_key=st.secrets["open_ai_key"], streaming=True) 
        qa_chain = load_qa_with_sources_chain(
            llm, chain_type='stuff',
            prompt=_qa_prompt,
            document_prompt=_doc_prompt
        )
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, input_key='question', output_key='answer', k=15)
        qa = RetrievalQAWithSourcesChain(
            combine_documents_chain=qa_chain, retriever=_db.as_retriever(),
            reduce_k_below_max_tokens=True, return_source_documents=True,
            memory=memory
        )
    return qa, memory

def format_names(name):
    if name == 'pt_example_db':
        return 'PT Example DB'
    else:
        return name.replace('_', ' ').title().replace('Db', 'DB')

def clean_cite(text):
    text = re.sub('\d+\:\d+\:\d+','',text)
    text = re.sub('\[.*?\]','',text)
    text = text.replace('cielo24 | what’s in your video? | cielo24.com', '')
    return text

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")