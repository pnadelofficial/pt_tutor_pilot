import streamlit as st
from langchain.vectorstores import FAISS
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import os
import pandas as pd
from prompts import prompt_dict
from feedback import FeedbackSurvey
import utils

st.title('Tufts Physical Therapy AI Tutor')

hf = utils.prep_embeddings()
model_choice = st.selectbox("What model would you like to use?", ["GPT 3.5", "GPT 4", ], help='Anthropic API access coming soon!')
doc_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"]
)
db_choice = st.selectbox("What week's material would you like to search?", os.listdir('./dbs'), format_func=utils.format_names)
db = FAISS.load_local(f'dbs/{db_choice}', hf)

template_radio = st.radio(
    "What would you like to do?",
    list(prompt_dict)+['Use my own'],
)
if template_radio in prompt_dict:
    template = prompt_dict[template_radio]
    qa_prompt = PromptTemplate.from_template(template)
else:
    template = st.text_area('Input your own prompt')
    qa_prompt = PromptTemplate.from_template(template)

qa, memory = utils.prep_model(model_choice, qa_prompt, doc_prompt, db)
msgs = StreamlitChatMessageHistory()

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if question := st.chat_input("Ask me anything!"):
    st.chat_message("user").write(question)
    msgs.add_user_message(question)

    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            result = qa({"question":question, "chat_history":memory.chat_memory})
        response = result['answer']
        cites = result['source_documents']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(response, unsafe_allow_html=True)
            msgs.add_ai_message(response)

        with col2: 
            with st.expander("**Citations**"):
                cite_df = pd.DataFrame(cites)
                cites_meta = list(zip(list(c[1] for c in cite_df.drop_duplicates(subset=0)[0]), list(c[1] for c in cite_df.drop_duplicates(subset=0)[1])))
                for cite in cites_meta:
                    st.markdown(f"*{cite[1]['source'].split('.pdf')[0]}*")
                    st.markdown(utils.clean_cite(cite[0]), unsafe_allow_html=True)           

with st.sidebar:
    st.write('# **Remember to give us your feedback!**\n')
    fs = FeedbackSurvey(msgs)
    fs.make_survey()