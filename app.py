import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from typing import Iterator
from agno.storage.sqlite import SqliteStorage
from agno.run.response import RunResponse
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

DEBUG_MODE = True

# Set page config
st.set_page_config(
    page_title="Agno Chat Interface",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_agent_team():
    web_agent = Agent(
        name="Web Agent",
        role="Search the web for information",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[DuckDuckGoTools()],
        instructions=["Always include sources", "But try to directly answer the question"],
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=DEBUG_MODE,
    )

    finance_agent = Agent(
        name="Finance Agent",
        role="Get financial data",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
        instructions="Use tables to display data",
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=DEBUG_MODE,
    )

    coding_agent = Agent(
        name="Coding Agent",
        role="Write and analyze code",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Write clean, efficient, and well-documented code",
            "Follow best practices and coding standards",
            "Include comments explaining complex logic",
            "Provide code examples with proper syntax highlighting",
            "Consider edge cases and error handling",
            "Suggest optimizations and improvements",
            "Explain your code decisions and trade-offs"
        ],
        show_tool_calls=True,
        markdown=True,
        debug_mode=DEBUG_MODE,
    )

    agent_storage: str = "tmp/agents.db"

    agent_team = Agent(
        team=[web_agent, finance_agent, coding_agent],
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=["Always include sources", "Use tables to display data"],
        show_tool_calls=True,
        debug_mode=DEBUG_MODE,
        # Store the agent sessions in a sqlite database
        storage=SqliteStorage(table_name="web_agent", db_file=agent_storage),
        # Adds the current date and time to the instructions
        add_datetime_to_instructions=True,
        # Adds the history of the conversation to the messages
        add_history_to_messages=True,
        # Number of history responses to add to the messages
        num_history_responses=5,
        # Adds markdown formatting to the messages
        markdown=True,
    )
    return agent_team

def parse_stream(stream):
    for chunk in stream:
        if chunk.content is not None:
            yield chunk.content

# App title and description
st.title("🤖 Agno Chat Interface")
st.caption("Chat with an intelligent team of agents.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "💁‍♀️"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar="💁‍♀️"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking..."):

            agent_team = get_agent_team()
            
            # agent_team.print_response(prompt, stream=True) # useful to just see the output in the terminal
            # Get the response
            stream: Iterator[RunResponse] = agent_team.run(
                prompt, 
                stream=True, 
                auto_invoke_tools=True,
                # stream_intermediate_steps=True, # useful to see the intermediate steps around the tool calls
            )
            response = st.write_stream(parse_stream(stream))

        st.session_state.messages.append({"role": "assistant", "content": full_response}) 