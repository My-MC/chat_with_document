import uuid
from pathlib import Path

import chromadb
import torch
import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DOCUMENT_DIR = "./docs"
PROMPT = "茨城県水戸市にある偕楽園で、梅を見ながら二匹のヒバリが出会う物語を作成してください。"  # noqa: E501


def fetch_document(dir: Path) -> list[Document]:
    documents = []

    for child in (path for path in dir.iterdir() if path.name != ".gitkeep"):
        contents = child.read_text().split("---")
        document = Document(
            page_content=contents[2], metadata=yaml.safe_load(contents[1])
        )

        documents.append(document)

    return documents


# 文書のチャンキング(512トークン単位)
text_splitter = MarkdownTextSplitter(chunk_size=512, chunk_overlap=50)
documents = text_splitter.split_documents(fetch_document(Path(DOCUMENT_DIR)))

# Granite Embeddingでベクトル化
model = SentenceTransformer("ibm-granite/granite-embedding-278m-multilingual")
embeddings = model.encode(
    [doc.page_content for doc in documents], batch_size=128, show_progress_bar=True
)


client = chromadb.Client()

collection = client.create_collection("document_store")

# メタデータとIDを生成
metadatas = [{"source": doc.metadata.get("source", "unknown")} for doc in documents]
ids = [str(uuid.uuid4()) for _ in documents]

# 埋め込みをChromaに保存
collection.add(
    embeddings=embeddings.tolist(),
    documents=[doc.page_content for doc in documents],
    metadatas=metadatas,
    ids=ids,
)


def retrieve_documents(query: str, k: int = 5) -> str:
    # クエリのベクトル化
    query_embedding = model.encode([query])

    # Chroma検索
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=k)

    return results["documents"]


def generate_prompt(query: str) -> str:
    context = retrieve_documents(query)

    prompt = f"""以下の文脈を参照し、質問に簡潔に回答してください:

{context}

質問:{query}"""

    return prompt


model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=1, quantization="bitsandbytes")

sampling_params = SamplingParams(
    temperature=0.6, top_p=0.9, max_tokens=8192, stop="<|eot_id|>"
)


message = [
    {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。"},
    {
        "role": "user",
        "content": generate_prompt(PROMPT),
    },
]
prompt = tokenizer.apply_chat_template(
    message, tokenize=False, add_generation_prompt=True
)

outputs = llm.generate(prompt, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

torch.distributed.destroy_process_group()
