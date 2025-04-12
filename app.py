import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from typing import Iterator
from agno.run.response import RunResponse

# Set page config
st.set_page_config(
    page_title="Agno Chat Interface",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the Agno agent
@st.cache_resource
def get_agent():
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        description="You are a helpful AI assistant that provides clear and concise responses.",
        markdown=True
    )

# App title and description
st.title("🤖 Agno Chat Interface")
st.markdown("Chat with an AI powered by Agno and GPT-4o-mini")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get the agent
        agent = get_agent()
        
        # Get the response
        run_response: Iterator[RunResponse] = agent.run(prompt, stream=True)
        for chunk in run_response:
            if chunk.content:
                full_response += str(chunk.content)
                message_placeholder.markdown(full_response + "▌")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response}) 