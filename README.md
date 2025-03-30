# Chat with Documents

Chat with LLM with some documents. This application is optimized for Japanese.

## How to Use

### Environment Setup

Setup Python Environment with [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

If you are in trouble while installing packages, please make sure that your shell is in clean virtual environment.

## Setup Information
1. Collect Markdown document, and put it into `./docs`
2. Change constant(`DOCUMENT_DIR`) in `main.py`.
3. Change rewrite prompt constant(`PROMPT`) in `main.py`

## Using Models

### [`Granite-Embedding-278m-multilingual`](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual)

Embedding model from IBM. This model is used for vectorize and searching collect topic from documents.

### [`Llama 3.1 Swallow`](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3)

Llama 3.1 pre-tuned model by tokyotech-llm. This model is main LLM of this application. This is why the application is optimized for Japanese.
