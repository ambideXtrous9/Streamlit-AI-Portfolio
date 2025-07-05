import streamlit as st
from NewsQALLM.HpAgent import graph

app = graph.compile()



def ChatBot():

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get the assistant's response using the predict function
        output = app.invoke({"topic": prompt, "review": "Write an awesome article on the topic."})
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(output["draft"])


        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": output["draft"]})
