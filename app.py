import streamlit as st
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from llama_index.indices.vector_store import VectorStoreIndex
import psycopg
from pgvector.psycopg import register_vector
from llama_index import Document
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index import set_global_service_context
from llama_index.embeddings import resolve_embed_model
from llama_index import ServiceContext
from llama_index.retrievers import QueryFusionRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.llama_pack import download_llama_pack

st.title("PT Tutor App")

@st.cache_resource
def get_index(llm='local'):
    db_name = "postgres"
    conn = psycopg.connect(dbname=db_name, host = "localhost", port = "5432", autocommit=True)
    conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(conn)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents;")
    rows = cursor.fetchall()
    docs_from_db = [Document(text=row[2],metadata={'source':row[1]},embedding=list(row[3])) for row in rows]
    
    if llm == 'openai': 
        index = VectorStoreIndex.from_documents(docs_from_db)
        return index
    else: 
        llm = LlamaCPP(model_path='/Volumes/T7 Shield/text-generation-webui/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf', temperature=.5)
        embed_model = resolve_embed_model("local:BAAI/bge-large-en")       
        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
        set_global_service_context(service_context)
        index = VectorStoreIndex.from_documents(docs_from_db)
        return index, service_context, llm

index, service_context, llm = get_index()
set_global_service_context(service_context)

## FUZZY CITATION AND QUERY REWRITE
vector_retriever = index.as_retriever(similarity_top_k=5)
fusion_retriever = QueryFusionRetriever(
    [vector_retriever],
    similarity_top_k=5,
    num_queries=6,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=False,
    verbose=True,
    llm=llm
    # query_gen_prompt="...",  # we could override the query generation prompt here
)
query_engine = RetrieverQueryEngine.from_args(fusion_retriever)
FuzzyCitationEnginePack = download_llama_pack("FuzzyCitationEnginePack", "./fuzzy_pack")
fuzzy_engine_pack = FuzzyCitationEnginePack(query_engine, threshold=50)

msgs = StreamlitChatMessageHistory()

if question := st.chat_input("Ask me anything!"):
    st.chat_message("user").write(question)
    msgs.add_user_message(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = fuzzy_engine_pack.run(question) #query_engine.query(question)
        response = str(result)
        cites = result.source_nodes

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(response, unsafe_allow_html=True)
            msgs.add_ai_message(response)
        
        with col2:
            with st.expander("**Citations**"):
                cites = [(c.metadata['source'], c.text) for c in cites]
                node_chunks = [node_chunk for _, node_chunk in result.metadata.keys()]
                for title, cite in cites:
                    for node_chunk in node_chunks:
                        if node_chunk in cite:
                            start_idx = cite.find(node_chunk)
                            end_idx = start_idx + len(node_chunk)
                            cite = f"""
                            <p>
                                {cite[:start_idx]}<mark style='background-color:#fdd835'>{cite[start_idx:end_idx]}</mark>{cite[end_idx:]}
                            </p>
                            """
                    st.markdown(f"*{title}*")
                    st.markdown(cite, unsafe_allow_html=True)