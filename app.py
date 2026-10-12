"""Local Ollama-powered e-commerce customer support chat application."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import httpx
import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from ollama import ResponseError

MODEL_NAME = "llama3.2:1b"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


class ChatTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


SYSTEM_PROMPT = """You are a helpful, empathetic, professional e-commerce customer
support assistant. Use clear, natural language. Answer the customer's question
directly with practical general guidance before asking anything. Do not give
one-line apologies in place of help, repeat the question, or request an order
number.

For a delayed order, suggest checking the confirmation email and carrier
tracking, checking the latest estimated delivery date, and contacting the
carrier or store through its official support channel if tracking is stalled or
the estimate has passed. For a damaged product, suggest stopping use, keeping
the item and packaging, taking photos if useful, and contacting the store's
official support channel. For a return, suggest reviewing the store's published
return policy and contacting the store to confirm the applicable steps. For a
refund, explain generally that the customer can contact the store and check its
published refund guidance; do not claim to check or process a refund.

Structure answers so they are easy to act on: begin with a brief
acknowledgement, then give a short explanation and numbered steps when there
are multiple actions. Usually keep steps to 2-4. Be complete without being
repetitive; do not restate the steps in another list or add irrelevant
alternatives. Use official store or carrier support channels for escalation;
do not suggest forums or social media. Ask a follow-up question only when the
answer truly depends on information not yet provided; when general advice is
possible, give it first. Do not add an unnecessary question at the end of an
answer. Use earlier user and assistant messages to understand references and
follow-up questions, and do not ask the customer to repeat information already
shared.

You have no access to orders, shipping systems, company-specific policies, or
payment/refund processing. Never invent order details, eligibility,
company-specific procedures, or timelines, and never claim you checked an
order, approved a return, or processed a transaction. If asked to verify an
order or refund, say: "Real order verification and refund processing require
integration with an actual business system." Do not request order numbers,
passwords, payment-card details, or other sensitive information."""


def check_ollama_model(
    base_url: str = OLLAMA_BASE_URL, model_name: str = MODEL_NAME
) -> tuple[bool, str]:
    """Check whether the local Ollama server is reachable and has the model."""
    try:
        with urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=3) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return (
            False,
            "Ollama is unavailable. Start Ollama and try again "
            "(for example, run `ollama serve`).",
        )

    model_entries = payload.get("models", []) if isinstance(payload, dict) else []
    available_models = {
        entry.get("name") or entry.get("model")
        for entry in model_entries
        if isinstance(entry, dict)
    }
    if model_name not in available_models:
        return (
            False,
            f"The {model_name} model is not installed. Install it with "
            f"`ollama pull {model_name}`.",
        )
    return True, "Ollama is ready."


def build_chain(llm: Runnable | None = None) -> Runnable:
    """Create a LangChain prompt/model/parser chain."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    model = llm if llm is not None else ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    return prompt | model | StrOutputParser()


def to_langchain_messages(history: Sequence[ChatTurn]) -> list[BaseMessage]:
    """Translate Streamlit's isolated session history to LangChain messages."""
    messages: list[BaseMessage] = []
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages


def answer_question(
    question: str,
    history: Sequence[ChatTurn],
    chain: Runnable,
) -> str:
    """Generate a response using previous turns and reject empty questions."""
    if not question.strip():
        raise ValueError("Enter a question before sending.")
    response = chain.invoke(
        {
            "history": to_langchain_messages(history),
            "question": question.strip(),
        }
    )
    if not isinstance(response, str):
        raise TypeError("The response was not text.")
    return response


def main() -> None:
    st.set_page_config(
        page_title="Customer Support Assistant",
        page_icon="🛍️",
        layout="centered",
    )
    st.markdown(
        """
        <style>
        :root {
            --support-ink: #17324d;
            --support-muted: #60758a;
            --support-teal: #0b8f87;
            --support-coral: #f08a70;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 8% 0%, #dff5f0 0, transparent 32rem),
                radial-gradient(ellipse at 100% 12%, #fff0e9 0, transparent 28rem),
                #f7fafc;
            color: var(--support-ink);
        }
        .block-container {
            max-width: 850px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }
        h1 {
            color: var(--support-ink);
            font-size: clamp(2rem, 5vw, 2.8rem) !important;
            font-weight: 750 !important;
            letter-spacing: -0.045em !important;
        }
        [data-testid="stCaptionContainer"] {
            color: var(--support-muted);
            font-size: 1.05rem;
            margin-bottom: 1.8rem;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid #e5edf2;
            border-radius: 1.15rem;
            box-shadow: 0 8px 24px rgba(23, 50, 77, 0.055);
            padding: 1rem 1.15rem;
            margin-bottom: 0.8rem;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #e5f4f2;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: #ffffff;
        }
        [data-testid="stChatInput"] {
            border: 1px solid #d7e3e9;
            border-radius: 1rem;
            box-shadow: 0 10px 28px rgba(23, 50, 77, 0.08);
            background: #ffffff;
        }
        [data-testid="stChatInput"] textarea {
            color: var(--support-ink);
            font-size: 1rem;
        }
        [data-testid="stChatInput"] button {
            color: #ffffff;
            background: var(--support-teal);
            border-radius: 0.7rem;
        }
        [data-testid="stChatInput"] button:hover {
            background: #08736e;
            color: #ffffff;
        }
        [data-testid="stAlert"] {
            border-radius: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Customer Support Assistant")
    st.caption("Your AI-powered assistant for e-commerce customer support.")

    is_ready, status_message = check_ollama_model()
    if not is_ready:
        st.warning(status_message)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for turn in st.session_state.messages:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask a customer support question")
    if question is None:
        return
    if not question.strip():
        st.error("Enter a question before sending.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not is_ready:
        st.error(status_message)
        return

    with st.chat_message("assistant"):
        with st.spinner("Preparing a response..."):
            try:
                response = answer_question(
                    question,
                    st.session_state.messages[:-1],
                    build_chain(),
                )
            except (
                HTTPError,
                URLError,
                TimeoutError,
                OSError,
                httpx.HTTPError,
                ResponseError,
            ):
                st.error(
                    "The local AI service could not complete the request. "
                    "Check that Ollama is running and try again."
                )
                return
            except (ValueError, TypeError):
                st.error("The question could not be processed. Please try again.")
                return

        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
