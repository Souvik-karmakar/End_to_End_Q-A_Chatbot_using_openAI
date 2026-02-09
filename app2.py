import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables (safe even if .env is missing)
load_dotenv()

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer clearly and concisely."),
        ("user", "{question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):
    # Set OpenAI key safely
    openai.api_key = api_key

    llm = ChatOpenAI(
        model=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,  # explicit & safe
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Q&A Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Q&A Chatbot")
st.caption("Ask anything. Powered by OpenAI + LangChain")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Your API key is never stored"
    )

    engine = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4-turbo", "gpt-4"]
    )

    temperature = st.slider(
        "Creativity (Temperature)",
        0.0, 1.0, 0.7
    )

    max_tokens = st.slider(
        "Max Tokens",
        50, 500, 200
    )

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
user_prompt = st.chat_input(
    "Type your question here...",
    disabled=not api_key
)

if user_prompt:
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            try:
                response = generate_response(
                    user_prompt,
                    api_key,
                    engine,
                    temperature,
                    max_tokens
                )
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ---------------- WARNINGS ----------------
if not api_key:
    st.info("🔑 Please enter your OpenAI API key in the sidebar to start chatting.")
