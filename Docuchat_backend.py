from pydantic import BaseModel
import pymongo
import traceback
from typing import Optional
from pathlib import Path

import os, sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.callbacks import get_openai_callback
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import gc

if os.name == "nt":  # Windows
    from dotenv import load_dotenv
    load_dotenv(".secrets.env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGO_URL = os.environ.get("MONGO_URL")

# Local folder the chatbot reads PDFs from. Drop reference docs here.
DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"

# S3 document source, currently DISABLED (bucket no longer exists).
# To re-enable: set USE_S3=true plus S3_KEY, S3_SECRET, S3_PATH env vars,
# and uncomment awswrangler in requirements.txt.
USE_S3 = os.environ.get("USE_S3", "false").lower() == "true"
S3_KEY = os.environ.get("S3_KEY")
S3_SECRET = os.environ.get("S3_SECRET")
S3_BUCKET = "sajans-website"
S3_PATH = os.environ.get("S3_PATH")


# Chats are saved here: database "chat_with_doc", collection "chat-history".
# Stays None if Mongo is unreachable, which the "/" health check reports.
conversationcol = None

try:
    print("OPENAI_API_KEY loaded:", bool(OPENAI_API_KEY))

    if not MONGO_URL:
        raise RuntimeError("MONGO_URL is not set")

    client = pymongo.MongoClient(
        MONGO_URL, uuidRepresentation="standard", serverSelectionTimeoutMS=5000
    )
    client.admin.command("ping")  # fail at startup, not on the first chat
    db = client["chat_with_doc"]
    conversationcol = db["chat-history"]
    conversationcol.create_index([("session_id", 1)], unique=True)
    print("MongoDB connected: chat_with_doc.chat-history")
except:
    conversationcol = None
    print("ERROR: MongoDB unavailable, chats will NOT be saved.")
    print(traceback.format_exc())
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)


def list_pdf_paths():
    """Return sorted list of PDF paths in DOCUMENTS_DIR."""
    if not DOCUMENTS_DIR.is_dir():
        return []
    return sorted(p for p in DOCUMENTS_DIR.glob("*.pdf") if p.is_file())


def load_local_docs():
    """Load every PDF in DOCUMENTS_DIR as LangChain documents."""
    pdf_paths = list_pdf_paths()
    if not pdf_paths:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF documents found in {DOCUMENTS_DIR}",
        )

    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        data = loader.load()
        for d in data:
            d.metadata["source"] = pdf_path.name
        all_docs.extend(data)
    return all_docs


def load_s3_docs():
    """Download every PDF under s3://S3_BUCKET/S3_PATH and load it. Only used when USE_S3=true."""
    # Imported here so awswrangler/boto3 are only required when S3 is enabled.
    import tempfile
    import boto3
    import awswrangler as wr

    aws_s3 = boto3.Session(
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET,
        region_name="us-east-2",
    )
    pdf_keys = wr.s3.list_objects(f"s3://{S3_BUCKET}/{S3_PATH}", boto3_session=aws_s3)
    pdf_keys = [k for k in pdf_keys if k.lower().endswith(".pdf")]
    if not pdf_keys:
        raise HTTPException(404, "No PDF documents found in S3 folder")

    all_docs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for s3_key in pdf_keys:
            file_name = os.path.basename(s3_key)
            local_path = os.path.join(tmpdir, file_name)
            s3_path = s3_key if s3_key.startswith("s3://") else f"s3://{S3_BUCKET}/{s3_key}"
            wr.s3.download(path=s3_path, local_file=local_path, boto3_session=aws_s3)

            loader = PyPDFLoader(local_path)
            data = loader.load()
            for d in data:
                d.metadata["source"] = file_name
            all_docs.extend(data)
    return all_docs


# Surface document-source state at startup so misconfig is obvious in logs.
_startup_pdfs = list_pdf_paths()
if USE_S3:
    print(f"Document source: S3 (s3://{S3_BUCKET}/{S3_PATH})")
elif not _startup_pdfs:
    print(
        f"WARNING: no PDFs found in {DOCUMENTS_DIR}. "
        "Drop reference documents into that folder for the chatbot to use."
    )
else:
    print("PDFs found:", [p.name for p in _startup_pdfs])


class ChatMessageSent(BaseModel):
    session_id: Optional[str] = None
    user_input: str


def get_response(
    session_id: str,
    query: str,
    model: str = "gpt-3.5-turbo-16k",
    temperature: float = 0.5,
):
    template = """
    You are an AI assistant representing Sajan Patel on his personal portfolio website.

    Your personality:
    - Friendly, warm, and conversational
    - Slightly expressive and energetic (but still professional)
    - Confident but not arrogant
    - Supportive and engaging

    Important rules:
    - Use ONLY the provided context to answer questions.
    - If the answer is not in the context, say you don't have that information.
    - Do NOT make up experience or skills.
    - Remember this is a portfolio website chatbot.

    IMPORTANT BEHAVIOR RULES:

    1. If a question asks about a programming language, framework, or technology
    that is NOT explicitly mentioned in the documents:

    - Do NOT say "I don't have that information."
    - Do NOT refuse the question.

    Instead:
    - Say you are not specifically versed in that language based on available info.
    - Pivot to related technologies Sajan DOES know.
    - Highlight transferable skills.

    Example:
    "While I don’t specifically list experience with Rust,
        I have strong backend experience with:
        - Python
        - API development
        - Database systems
        - Scalable architectures"

    2. If the question truly has no connection to Sajan or his work,
    politely redirect the conversation back to his skills and experience.

    Always answer confidently but honestly.
    Never fabricate experience.

    When asked about a technology not mentioned, infer related skills based on domain similarity.
    Example:
    - JavaScript → mention frontend architecture knowledge
    - React → mention component-based design experience
    - AWS service → mention cloud + infrastructure knowledge



    When appropriate, you can:
    - Show enthusiasm about projects
    - Highlight impact and problem-solving
    - Speak naturally instead of robotic

    RESPONSE STYLE RULES (VERY IMPORTANT):
    - Keep answers SHORT and concise
    - Use bullet points whenever possible
    - Each bullet point MUST be on its own line
    - Add a blank line before and after bullet lists
    - Never write multiple bullet points in a single sentence
    - Use spacing between sections
    -Format your response using proper Markdown syntax.
    - Use code blocks for technical examples
    - Avoid long paragraphs
    - Be clear and easy to skim
    - If answer is longer than 6 lines, restructure it
    - Maximum 120 words unless user explicitly asks for detailed explanation


    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}

    Answer:
    """

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    all_docs = load_s3_docs() if USE_S3 else load_local_docs()

    print("splitting ..")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=["\n", " ", ""]
    )
    all_splits = text_splitter.split_documents(all_docs)
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    llm = ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=OPENAI_API_KEY)

    QA_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question", "chat_history"],
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    with get_openai_callback() as cb:
        answer = qa_chain(
            {
                "question": query,
                "chat_history": load_memory_to_pass(session_id=session_id),
            }
        )
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        answer["total_tokens_used"] = cb.total_tokens
    gc.collect()
    return answer


import uuid
from datetime import datetime, timezone
from typing import List


def load_memory_to_pass(session_id: str):
    data = conversationcol.find_one({"session_id": session_id})
    history = []
    if data:
        data = data["conversation"]
        for x in range(0, len(data), 2):
            history.extend([(data[x], data[x + 1])])
    print(history)
    return history


def get_session() -> str:
    return str(uuid.uuid4())


def add_session_history(session_id: str, new_values: List):
    # Single atomic upsert: creates the session doc on first message, appends after.
    now = datetime.now(timezone.utc)
    conversationcol.update_one(
        {"session_id": session_id},
        {
            "$push": {"conversation": {"$each": new_values}},
            "$set": {"updated_at": now},
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sajan-patel.com"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def create_chat_message(chats: ChatMessageSent):
    try:
        if chats.session_id is None:
            session_id = get_session()
            payload = ChatMessageSent(
                session_id=session_id,
                user_input=chats.user_input,
            ).model_dump()

            response = get_response(
                session_id=payload.get("session_id"),
                query=payload.get("user_input"),
            )

            add_session_history(
                session_id=session_id,
                new_values=[payload.get("user_input"), response["answer"]],
            )

            return JSONResponse(
                content={"response": response, "session_id": str(session_id)}
            )

        else:
            payload = ChatMessageSent(
                session_id=str(chats.session_id),
                user_input=chats.user_input,
            ).dict()

            response = get_response(
                session_id=payload.get("session_id"),
                query=payload.get("user_input"),
            )

            add_session_history(
                session_id=str(chats.session_id),
                new_values=[payload.get("user_input"), response["answer"]],
            )

            return JSONResponse(
                content={"response": response, "session_id": str(chats.session_id)}
            )
    except Exception as e:
        print(traceback.format_exc())
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        # Full details are in the server logs above; don't expose internals to visitors.
        return JSONResponse(
            status_code=500, content={"error": "The assistant could not answer right now."}
        )


@app.get("/")
def health():
    return {"status": "alive", "mongo": conversationcol is not None}


@app.get("/documents")
async def get_available_documents():
    """List PDFs the chatbot is currently using as context."""
    try:
        documents = [
            {"name": p.name, "path": str(p.relative_to(DOCUMENTS_DIR)), "size": p.stat().st_size}
            for p in list_pdf_paths()
        ]
        return JSONResponse(content={"documents": documents})
    except Exception as e:
        print(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")
