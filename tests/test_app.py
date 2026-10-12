from io import BytesIO
import unittest
from urllib.error import URLError
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from app import (
    answer_question,
    build_chain,
    check_ollama_model,
    to_langchain_messages,
)


class CustomerSupportAssistantTests(unittest.TestCase):
    def test_history_is_converted_to_langchain_messages(self) -> None:
        history = [
            {"role": "user", "content": "My order is late."},
            {"role": "assistant", "content": "I can offer general guidance."},
        ]

        messages = to_langchain_messages(history)

        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)
        self.assertEqual(messages[0].content, "My order is late.")
        self.assertEqual(messages[1].content, "I can offer general guidance.")

    def test_chain_uses_history_and_returns_mocked_response(self) -> None:
        observed_messages = []

        def fake_model(prompt_value):
            observed_messages.extend(prompt_value.to_messages())
            return AIMessage(content="Please contact the store's support team.")

        chain = build_chain(RunnableLambda(fake_model))
        answer = answer_question(
            "Can you check my refund?",
            [{"role": "user", "content": "My product was damaged."}],
            chain,
        )

        self.assertEqual(answer, "Please contact the store's support team.")
        self.assertEqual(observed_messages[-2].content, "My product was damaged.")
        self.assertEqual(observed_messages[-1].content, "Can you check my refund?")
        self.assertIn("Never invent", observed_messages[0].content)
        self.assertIn(
            "practical general guidance before asking anything",
            observed_messages[0].content,
        )
        self.assertIn("Do not give", observed_messages[0].content)
        self.assertIn("checking the confirmation email", observed_messages[0].content)
        self.assertIn("stopping use", observed_messages[0].content)
        self.assertIn("return policy", observed_messages[0].content)
        self.assertIn("published refund guidance", observed_messages[0].content)
        self.assertIn(
            "integration with an actual business",
            observed_messages[0].content,
        )

    def test_follow_up_receives_full_previous_conversation(self) -> None:
        prompts = []

        def fake_model(prompt_value):
            messages = prompt_value.to_messages()
            prompts.append(messages)
            return AIMessage(content="I can provide general guidance.")

        chain = build_chain(RunnableLambda(fake_model))
        history = [
            {"role": "user", "content": "My order arrived damaged."},
            {
                "role": "assistant",
                "content": "Please stop using it and contact the store.",
            },
        ]

        answer = answer_question("Can I return it?", history, chain)

        self.assertEqual(answer, "I can provide general guidance.")
        self.assertEqual(
            [(message.type, message.content) for message in prompts[0][1:]],
            [
                ("human", "My order arrived damaged."),
                ("ai", "Please stop using it and contact the store."),
                ("human", "Can I return it?"),
            ],
        )

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Enter a question"):
            answer_question("  ", [], build_chain(RunnableLambda(lambda _: "")))

    def test_status_reports_missing_model_without_internal_details(self) -> None:
        response = BytesIO(b'{"models": []}')
        with patch("app.urlopen", return_value=response):
            is_ready, message = check_ollama_model("http://localhost:11434")

        self.assertFalse(is_ready)
        self.assertIn("ollama pull llama3.2:1b", message)

    def test_status_hides_connection_details(self) -> None:
        with patch("app.urlopen", side_effect=URLError("private connection detail")):
            is_ready, message = check_ollama_model("http://localhost:11434")

        self.assertFalse(is_ready)
        self.assertIn("Ollama is unavailable", message)
        self.assertNotIn("private connection detail", message)


if __name__ == "__main__":
    unittest.main()
