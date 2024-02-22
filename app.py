import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import utils
from PIL import Image

st.set_page_config("Dochat",":robot:","wide")
def initialize_session_state():
    """
    Session State is a way to share variables between reruns, for each user session.
    """

    st.session_state.setdefault('history',[])
    st.session_state.setdefault('generated',["Hello there!👋 Let's start a Question Answering session."])
    st.session_state.setdefault('past',["Hey!"])

def create_conversational_chain(llm, vector_store):
    """
    Creating conversational chain using Mistral 7B LLM instance and vector store instance

    Args:
    - llm: Instance of Mistral 7B GGUF
    - vector_store: Instance of FAISS Vector store having all the PDF document chunks
    """

    memory = ConversationBufferMemory(memory_key="chat_history", return_message=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                                   retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                                  memory=memory)
    return chain

def display_chat(conversation_chain):
    """
    Streamlit related code where we are passing conversation_chain instance created earlier
    It creates two containers
    container: To group our chat inputs form
    reply_container: To group the generated chat response

    Args:
     - conversation_chain: Instance of LangChain ConversationalRetrievalChain
    """
    reply_container = st.container()
    container = st.container()
    with container:
        with st.form(key='chat_form',clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Type your query here.")
            submit_button = st.form_submit_button(label="Send ▶")

        # Check if user submit question with user input and generate response of the question
        if submit_button and user_input:
            generate_response(user_input, conversation_chain)

    # Display generated response to streamlit web UI
    display_generated_response(reply_container)

def generate_response(user_input, conversation_chain):
    """
    Generate LLM response based on the user question by retrieving data from Vector Database
    Also, stores information to streamlit session states 'past' and 'generate' so that it can
    have memory of previous generation fo conversational type of chats

    Args:
    - user_input(str): User input as a text
    - conversation_chain: Instance of ConversationalRetrievalChain
    """

    with st.spinner("Getting your answer ready..."):
        output = conversation_chat(user_input, conversation_chain, st.session_state['history'])

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

def conversation_chat(user_input, conversation_chain, history):
    """
    Returns LLM response after invoking model through conversation_chain

    Args:
    - user_input(str): User input
    - conversation_chain: Instance of ConversationalRetrievalChain
    - history: Previous response history
    returns:
    - result["answer"]: Response generated from LLM
    """
    result = conversation_chain.invoke({"question": user_input, "chat_history": history})
    history.append((user_input, result["answer"]))
    return result["answer"]

def display_generated_response(reply_container):
    """
    Display generated LLM response to Streamlit Web UI

    Args:
    - reply_container: Streamlit container created at previous step
    """
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="adventurer")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

def main():
    """
    First function to call when we start streamlit app
    """
    # Step 1: Initialize session state
    initialize_session_state()

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    title = """
            <h1 style=position: fixed; text-align: center; color: #000000;> Dochat </h1>
            <h6 style=position: fixed; justify-content: center; color: #000000;> I'm a chatbuddy to make a little chat with your pdf files. </h6>
            """
    st.markdown(title, unsafe_allow_html=True)

    # Step 2: Initialize Streamlit
    image = Image.open("dochat.png")
    st.sidebar.image(image, width=250)
    st.sidebar.title("Upload your PDF files📄")
    # file_uploader, the data are copied to the Streamlit backend via the browser,
    # and contained in a BytesIO buffer in Python memory (i.e. RAM, not disk).
    pdf_files = st.sidebar.file_uploader("",accept_multiple_files=True)

    # Step 3: Create instance of Mistral 7B GGUF file format using llama.cpp
    llm = utils.create_llm()

    # Step 4: Create Vector store and store uploaded Pdf file to in-memory Vector Database FAISS
    # and return instance of vector store
    vector_store = utils.create_vector_store(pdf_files)

    if vector_store:
        # Step 5: If Vector Store created successfully with chunks of PDF files
        # then create the chain object
        chain = create_conversational_chain(llm, vector_store)

        # Step 6: Display Chat to Web UI
        display_chat(chain)
    else:
        print("Initialized App.")

if __name__ == "__main__":
    main()