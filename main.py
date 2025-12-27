from fastapi import FastAPI
from pydantic import BaseModel
import json
import os

from fastapi.middleware.cors import CORSMiddleware

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms import HuggingFacePipeline

from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline


# -------------------- FASTAPI --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- CHAT HISTORY STORE --------------------
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# -------------------- LOAD FAQ DATA --------------------
with open("qa.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

docs = [
    Document(page_content=f"Q: {item['question']}\nA: {item['answer']}")
    for item in faq_data
]


# -------------------- EMBEDDINGS + FAISS --------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if os.path.exists("faiss_index"):
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")


# -------------------- LLM --------------------
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_length=96
)

llm = HuggingFacePipeline(pipeline=pipe)


# -------------------- RETRIEVER --------------------
retriever = db.as_retriever(search_kwargs={"k": 1})


# -------------------- PROMPT --------------------
prompt = PromptTemplate.from_template(
    """You are a helpful assistant.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:"""
)


# -------------------- RAG CHAIN (CORRECT) --------------------
rag_chain = (
    {
        "question": lambda x: x["question"],
        "context": lambda x: retriever.invoke(x["question"]),
        "chat_history": lambda x: x.get("chat_history", []),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# -------------------- ADD MESSAGE HISTORY --------------------
rag_chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


# -------------------- API MODEL --------------------
class Query(BaseModel):
    question: str
    session_id: str = "default_user"


# -------------------- UTILITY --------------------
def clean_text(text: str) -> str:
    return (
        text.replace("â€™", "'")
            .replace("â€TM", "'")
            .replace("â€œ", '"')
            .replace("â€�", '"')
    )

def clean_rag_output(output):
    """
    Ensure output is a plain string without extra list brackets or weird characters.
    """
    if isinstance(output, list):
        # Flatten list to single string
        output = " ".join([str(o) for o in output])
    # Replace weird Unicode issues from LLM (common with HF Flan-T5)
    output = output.replace("â€™", "'").replace("â€TM", "'")
    # Strip any leading/trailing whitespace or dangling brackets
    output = output.strip().rstrip("')]")
    return output

# -------------------- API ENDPOINT --------------------
@app.post("/chat")
def chat(query: Query):
    try:
        raw_result = rag_chain_with_history.invoke(
            {"question": query.question},
            config={"configurable": {"session_id": query.session_id}}
        )

        # Clean it properly
        answer = clean_rag_output(raw_result)

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}
