import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

import os
from langchain.tools import BaseTool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from duckduckgo_search import DuckDuckGoSearchException


class SafeDuckDuckGoSearch(BaseTool):
    name = "Search"
    description = (
        "Use this to search the web (DuckDuckGo). "
        "Returns short text snippets from the results."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use backend='html' or 'lite' to reduce issues with the API backend
        self.api_wrapper = DuckDuckGoSearchAPIWrapper(
            backend="html",   # options: "api", "html", "lite"  [oai_citation:0‡aidoczh.com](https://www.aidoczh.com/langchain_api/html/_modules/langchain_community/utilities/duckduckgo_search.html?utm_source=chatgpt.com)
            max_results=5,
        )

    def _run(self, query: str, run_manager=None) -> str:
        try:
            return self.api_wrapper.run(query)
        except DuckDuckGoSearchException as e:
            # swallow the exception and return a friendly message to the agent
            return (
                "I tried to search the web but DuckDuckGo is rate-limiting requests "
                "from this environment. I couldn't fetch live results right now."
            )

    async def _arun(self, query: str, run_manager=None) -> str:
        # you can implement async if needed later
        return self._run(query, run_manager=run_manager)

from dotenv import load_dotenv

load_dotenv()

# --- Tools setup ---

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# --- Streamlit UI ---

st.title("🔎 LangChain - Chat with search")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can search the web. How can I help you?",
        }
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        # LLM
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            streaming=True,
        )

        tools = [search, arxiv, wiki]

        # Callback for nice tool traces in Streamlit
        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False,
        )

        # Agent – ZERO_SHOT_REACT_DESCRIPTION
        search_agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,  # ✅ correct arg name
            verbose=True,
        )

        with st.chat_message("assistant"):
            # Run ONLY on the latest user prompt, not the whole messages list
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state["messages"].append(
                {"role": "assistant", "content": response}
            )
            st.write(response)
