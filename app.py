import streamlit as st
from dotenv import load_dotenv, find_dotenv
import os
import cassio
from langchain.memory import CassandraChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain import PromptTemplate

load_dotenv(find_dotenv(), override=True)

cassio.init(token=os.environ["ASTRA_DB_APPLICATION_TOKEN"], database_id=os.environ["ASTRA_DB_ID"])



# Globals
cqlMode = 'astra_db'  # 'astra_db'/'local'
session = cassio.config.resolve_session()
keyspace = cassio.config.resolve_keyspace()
memory_table_name = 'astra_agent_memory'
kb_table_name = 'vs_investment'
llm = OpenAI(temperature=0.5)
embedding_generator = OpenAIEmbeddings()

CassVectorStore = Cassandra(
    session= cassio.config.resolve_session(),
    keyspace= 'demo',
    table_name= kb_table_name,
    embedding=embedding_generator
)

index = VectorStoreIndexWrapper(
    vectorstore=CassVectorStore
)

def clear_memory(conversation_id):
    message_history = CassandraChatMessageHistory(
        session_id=conversation_id,
        session=session,
        keyspace=keyspace,
        ttl_seconds=3600,
    )

    message_history.clear()
    del st.session_state['conversation_id']
    del st.session_state['messages']
    del st.session_state['summary']
    return True

def start_memory():
    load_memory(st.session_state["conv_id_input"])
    return True



def get_answer(conversation_id, q):
    prompt_template = """
    Given the following extracted parts of a long document and a question, create a final answer in a very short format. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    Answer in Portuguese.


    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["summaries", "question"]
    )

    message_history = CassandraChatMessageHistory(
        session_id=conversation_id,
        session= cassio.config.resolve_session(),
        keyspace= cassio.config.resolve_keyspace(),
        ttl_seconds=3600,
        table_name=memory_table_name
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=message_history,
        max_token_limit=50,
        buffer=""
    )

    retrieverSim = CassVectorStore.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            'k': 5,
            'filter': {"source": "./funds/RealInvFIM0623.pdf"},
            "score_threshold": .8
        },
    )
    
    # Create a "RetrievalQA" chain
    chainSim = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retrieverSim,
        memory=memory,
        chain_type_kwargs={
            'prompt': PROMPT,
            'document_variable_name': 'summaries'
        }
    )
    new_summary = memory.predict_new_summary(
        memory.chat_memory.messages,
        memory.moving_summary_buffer,
    )

    st.session_state.messages = memory.chat_memory.messages
    st.session_state.summary = new_summary
    
    # Run it and print results
    answer = chainSim.run(q)

    return answer



def get_answer_2(conversation_id, q):
    st.session_state.conversation_id = conversation_id

    prompt_template = """
    Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer. Answer in Portuguese.


    QUESTION: {question}
    =========
    PREVIOUS CONVERSATION: {summary}
    =========
    {summaries}
    =========
    FINAL ANSWER:"""

    message_history = CassandraChatMessageHistory(
        session_id=conversation_id,
        session=session,
        keyspace=keyspace,
        ttl_seconds=3600,
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=message_history,
        max_token_limit=180,
        buffer=""
    )

    summaryConversation = ConversationChain(
        llm=llm,
        memory=memory

    )

    answer = summaryConversation.predict(input=q)
    print("Full answer")
    print(answer)

    new_summary = memory.predict_new_summary(
        memory.chat_memory.messages,
        memory.moving_summary_buffer,
    )

    st.session_state.messages = memory.chat_memory.messages
    st.session_state.summary = new_summary

    return answer


def load_memory(conversation_id):
    st.session_state.conversation_id = conversation_id

    message_history = CassandraChatMessageHistory(
        session_id=conversation_id,
        session=session,
        keyspace=keyspace,
        ttl_seconds=3600,
        table_name=memory_table_name
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=message_history,
        max_token_limit=180,
        buffer=""
    )

    new_summary = memory.predict_new_summary(
        memory.chat_memory.messages,
        memory.moving_summary_buffer,
    )

    st.session_state.messages = memory.chat_memory.messages
    st.session_state.summary = new_summary

    return memory.chat_memory.messages, new_summary


def format_messages(messages):
    res = ""
    for m in reversed(messages):
        res += f'{type(m).__name__}: {m.content}\n\n'
    return res


if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    import os
    load_dotenv(find_dotenv(), override=True)

    st.subheader('Recomendação de investimentos com IA Generativa e Astra')
    with st.sidebar:
        conversation_id = st.text_input(
            'Conversation ID', 'my-conv-id-01')
        clear_data = st.button(
            'Clear History', on_click=clear_memory, args=[conversation_id])
        load_data = st.button(
            'Load Conversation Memory', on_click=load_memory, args=[conversation_id])

    q = st.text_input("Message")
    if q:
        answer = get_answer(conversation_id, q)
        st.text_area('LLM Answer: ', value=answer)

    if 'summary' in st.session_state:
        st.divider()
        st.text_area(label=f"Summary for conversation id: {st.session_state.conversation_id}", value=st.session_state.summary, height=200)
    
    if 'messages' in st.session_state:
        st.divider()
        st.text_area(label="Memory", value=format_messages(
            st.session_state.messages), height=400)
