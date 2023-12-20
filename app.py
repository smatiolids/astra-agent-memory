import streamlit as st
from dotenv import load_dotenv, find_dotenv
import os
import cassio
from langchain.memory import AstraDBChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AstraDB
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv(), override=True)

# Globals

memory_table_name = 'vs_investment_memory'
kb_table_name = 'vs_investment_kb'

llm = OpenAI(temperature=0.1)
embedding_generator = OpenAIEmbeddings()

AstraVectorStore = AstraDB(
    embedding=embedding_generator,
    collection_name=kb_table_name,
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
)

index = VectorStoreIndexWrapper(
    vectorstore=AstraVectorStore
)


def clear_memory(conversation_id):
    message_history = AstraDBChatMessageHistory(
        session_id=conversation_id,
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        # ttl_seconds=3600,
        collection_name=memory_table_name
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

    message_history = AstraDBChatMessageHistory(
        session_id=conversation_id,
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        # ttl_seconds=3600,
        collection_name=memory_table_name
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=message_history,
        max_token_limit=50,
        buffer=""
    )

    # retrieverSim = AstraVectorStore.as_retriever(
    #     search_type='similarity_score_threshold',
    #     search_kwargs={
    #         'k': 5,
    #         'filter': {"source": st.session_state.file},
    #         "score_threshold": .8
    #     },
    # )

    retrieverSim = AstraVectorStore.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': 5,
            'filter': {"source": st.session_state.file}
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


def load_memory(conversation_id, file):
    st.session_state.conversation_id = conversation_id
    st.session_state.file = file

    message_history = AstraDBChatMessageHistory(
        session_id=conversation_id,
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        # ttl_seconds=3600,
        collection_name=memory_table_name
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

    st.subheader('Agente de investimentos com IA Generativa e Astra')
    with st.sidebar:
        conversation_id = st.text_input(
            'Conversation ID', 'my-conv-id-01')
        file = st.text_input(
            'File', './pdf/Lamina_12082452000149_v46.pdf')
        # clear_data = st.button(
        #     'Clear History', on_click=clear_memory, args=[conversation_id])
        load_data = st.button(
            'Load Conversation Memory', on_click=load_memory, args=[conversation_id, file])

    q = st.text_input("Message")
    if q:
        answer = get_answer(conversation_id, q)
        st.text_area('LLM Answer: ', value=answer)

    if 'summary' in st.session_state:
        st.divider()
        st.text_area(
            label=f"Summary for conversation id: {st.session_state.conversation_id}", value=st.session_state.summary, height=200)

    if 'messages' in st.session_state:
        st.divider()
        st.text_area(label="Memory", value=format_messages(
            st.session_state.messages), height=400)
