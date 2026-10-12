# AI Customer Support Assistant Using LangChain

A local Streamlit chatbot for common e-commerce support questions. LangChain
assembles a guarded support prompt, conversation history, and a local Ollama
chat model. All inference stays on the machine running Ollama; the app does not
use paid APIs, API keys, or hosted LLM services.

## Architecture

- `app.py` renders the Streamlit chat UI and keeps each browser session's
  conversation in that session's `st.session_state`.
- A LangChain `ChatPromptTemplate` combines the support instructions,
  prior user/assistant messages, and the current question.
- `ChatOllama` connects to the local Ollama server using `llama3.2:1b`;
  `StrOutputParser` returns the assistant's text.
- The support prompt helps with delayed orders, damaged products, returns,
  and refunds. It prohibits invented order details, policies, and statuses.
  This training/demo assistant cannot verify orders or process refunds; those
  tasks require integration with a real business system.

## Requirements

- Linux, macOS, or Windows with WSL2
- Python 3.10 or newer
- Ollama installed and running
- Approximately 2 GB or more of memory available to run the lightweight model;
  additional disk space is needed for the model download

## Setup

1. Install Ollama using the instructions for your operating system at
   [ollama.com/download](https://ollama.com/download). On Linux, the official
   installation command is `curl -fsSL https://ollama.com/install.sh | sh`;
   it may request administrator privileges.
2. Start Ollama if it is not already running. On Linux or in a terminal, use
   `ollama serve`. Some desktop installations start it automatically.

   In the Codespaces environment where this project was initialized, the
   default filesystem did not have enough room to finish model installation.
   Its model is stored on the available `/vscode` mount; after restarting that
   environment, start the service with:

   ```bash
   OLLAMA_MODELS=/vscode/ollama/models ollama serve
   ```

   On other machines, use the normal Ollama startup command and default model
   storage.
3. Download the configured model once:

   ```bash
   ollama pull llama3.2:1b
   ```

4. Create and activate a project virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   `requirements.txt` also retains the dependencies used by the existing
   notebooks in this repository.

## Run

With the virtual environment active and Ollama running:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. Conversation state is stored per
Streamlit session and is not shared between separate sessions.

## Tests

Run the mocked unit tests without making any model requests:

```bash
python -m unittest discover -s tests
```

To check that the installed model responds directly through Ollama:

```bash
ollama run llama3.2:1b "Reply with a short greeting."
```

## Troubleshooting

- **Ollama is unavailable:** start it with `ollama serve`, then reload the app.
  The server is expected at `http://localhost:11434`.
- **The model is not installed:** run `ollama pull llama3.2:1b`.
- **Model response fails:** check `ollama list`, confirm available disk and
  memory, and test the model with the `ollama run` command above.
- **The app cannot import a dependency:** activate `.venv` and rerun
  `python -m pip install -r requirements.txt`.

The app reports understandable setup and connection errors and does not fall
back to an external model provider.
