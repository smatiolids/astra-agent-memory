import streamlit as st
from langchain.memory import CassandraChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

from cqlsession import getCQLSession, getCQLKeyspace
from langchain.llms import OpenAI


# Globals
cqlMode = 'astra_db'  # 'astra_db'/'local'
session = getCQLSession(mode=cqlMode)
keyspace = getCQLKeyspace(mode=cqlMode)
table_name = 'astra_agent_memory'
llm = OpenAI()


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
    st.session_state.conversation_id = conversation_id

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
        table_name='my_memory_table'
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

    st.subheader('Agent Memory with Astra')
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
