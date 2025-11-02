import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import UnstructuredFileLoader # Using Unstructured for all structured formats
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

import torch
from importlib.metadata import PackageNotFoundError 


EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
#LLM_MODEL_NAME = "Qwen/Qwen1.5-7B-Chat" 
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B"

global_retriever: Any = None
global_embed_model: Optional[HuggingFaceEmbeddings] = None
global_vector_store: Optional[FAISS] = None
global_llm: Optional[HuggingFacePipeline] = None
global_tokenizer: Any = None 

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_DATA_FOLDER = SCRIPT_DIR / "data"
DEFAULT_STORE_PATH = SCRIPT_DIR / "vector_store.faiss"

RAG_SYSTEM_PROMPT = """
Using the information contained in the context and conversation history, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
If the answer cannot be deduced from the context, do not give an answer.
"""

UNSTRUCTURED_EXTENSIONS = [
    ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".tsv", ".json", ".xml", # Documents and Data
    ".html", ".eml", ".msg", # Web and Email
    ".odt", ".ods", ".odp", # OpenDocument formats
    ".py", ".java", ".cpp", ".js", ".ts", ".md", ".css", ".sql", ".log", # Code and Config
]


def load_embed_model(embed_model_name: str = EMBED_MODEL_NAME) -> HuggingFaceEmbeddings:
    MODEL_CACHE = SCRIPT_DIR / "models"
    EMBED_ROOT_CACHE = MODEL_CACHE / "embedding_models"
    EMBED_ROOT_CACHE.mkdir(parents=True, exist_ok=True) 
    
    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        cache_folder=str(EMBED_ROOT_CACHE)
    )
    return embed_model

def load_plain_text_file(file_path: str) -> List[Document]:
    try:
        # Attempt to read as raw text, assuming UTF-8 encoding
        text = Path(file_path).read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError) as e:
        print(f"Warning: Skipping file {file_path} because it could not be read as UTF-8 text: {e}")
        return []
    metadata = {"source": str(file_path)}
    return [Document(page_content=text, metadata=metadata)]

def load_documents(folder: Path) -> List[Document]:
    docs = []
    if not folder.exists():
        return []

    for p in folder.rglob("*"):
        if p.is_file():
            suffix = p.suffix.lower()
            file_path = str(p)
            
            # --- Primary Loading Path (Explicitly Supported Types) ---
            if suffix in UNSTRUCTURED_EXTENSIONS:
                try:
                    print(f"Loading supported file with UnstructuredFileLoader: {p.name}")
                    loader = UnstructuredFileLoader(file_path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Error loading structured file {p}: {e}. Falling back to raw text loader.")
                    # Fallback to raw text loader if Unstructured fails
                    docs.extend(load_plain_text_file(file_path)) 
            # --- Fallback Path (Unknown/Generic Types) ---
            else:
                # Treat all other files as raw text fallback (config, logs, truly unknown formats)
                print(f"Loading unsupported file as raw text: {p.name}")
                docs.extend(load_plain_text_file(file_path))
            
    return docs

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_out = splitter.split_documents(documents)
    return docs_out

def build_vector_store(documents: List[Document], embed_model: HuggingFaceEmbeddings, persist_path: Path) -> FAISS:
    
    if persist_path.is_dir():
        shutil.rmtree(persist_path)
        
    vector_store = FAISS.from_documents(documents, embed_model)
    vector_store.save_local(str(persist_path))
    return vector_store

def load_vector_store(persist_path: Path, embed_model: HuggingFaceEmbeddings) -> FAISS:
    if not (persist_path.exists() and persist_path.is_dir()):
        raise FileNotFoundError(f"Vector store not found at {persist_path}.")
    
    return FAISS.load_local(str(persist_path), embed_model, allow_dangerous_deserialization=True)

def load_local_llm(model_name: str = LLM_MODEL_NAME) -> HuggingFacePipeline:
    global global_tokenizer 
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    MODEL_CACHE = SCRIPT_DIR / "models"
    LLM_ROOT_CACHE = MODEL_CACHE / "llm_models"
    LLM_ROOT_CACHE.mkdir(parents=True, exist_ok=True) 
    LLM_CACHE_PATH_STR = str(LLM_ROOT_CACHE) 

    print(f"Loading LLM: {model_name} on device: {DEVICE.upper()}")

    model_kwargs = {}
    
    try:
        print("Attempting to load model with 4-bit quantization (requires bitsandbytes).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config
        
        if DEVICE.startswith("cuda"):
            model_kwargs['device_map'] = 'auto'
        print("Quantization configured successfully.")

    except (PackageNotFoundError, ImportError, ValueError) as e:
        print(f"WARNING: Quantization failed. Model will be loaded in full precision on {DEVICE.upper()}. Error: {e}")
        if DEVICE.startswith("cuda"):
            model_kwargs['device_map'] = 'auto'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=LLM_CACHE_PATH_STR)
    global_tokenizer = tokenizer 

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=LLM_CACHE_PATH_STR, 
        **model_kwargs,
        trust_remote_code=True 
    )

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        return_full_text=False, 
        max_new_tokens=2048, 
        do_sample=True,
        temperature=0.3, 
        repetition_penalty=1.15,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def get_retriever(vector_store: FAISS, retriever_top_k: int = 10) -> Any:
    retriever = vector_store.as_retriever(search_kwargs={"k": retriever_top_k})
    return retriever


def assemble_qwen_prompt(
    current_query: str, 
    context_text: str, 
    history: List[Dict[str, str]], 
    system_prompt: str, 
    tokenizer: Any
) -> str:
    
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt.strip()},
    ]
    
    for message in history:
        role = message.get('role')
        content = message.get('content', '').strip()
        if role in ['user', 'assistant'] and content:
            messages.append({"role": role, "content": content})

    final_user_content = f"""
Context:
{context_text}
---
Now here is the question you need to answer.

Question: {current_query}
"""
    messages.append({"role": "user", "content": final_user_content.strip()})
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return prompt


app = FastAPI(
    title="Local RAG Document Indexer & Query Server (Qwen 7B Conversational)",
    description="A FastAPI server using the Qwen 7B Chat model with 4-bit quantization and chat history support.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.on_event("startup")
async def load_models_and_retriever():
    global global_retriever, global_embed_model, global_llm, global_vector_store
    
    print("Starting RAG Server: Initializing Models")

    try:
        global_embed_model = load_embed_model(EMBED_MODEL_NAME)
        print("Embedding Model loaded successfully.")

        # Force re-initialization: Delete previous store if it exists (User Request)
        if DEFAULT_STORE_PATH.is_dir():
            shutil.rmtree(DEFAULT_STORE_PATH)
            print("Existing Vector Store deleted successfully.")
            
        # Always create a fresh placeholder vector store to ensure the retriever is functional
        print("Creating a fresh placeholder vector store for initialization.")
        dummy_doc = [Document(page_content="This is a placeholder for an empty knowledge base. Please index documents using the /index endpoint.", metadata={"source": "Internal System"})]
        global_vector_store = build_vector_store(dummy_doc, global_embed_model, DEFAULT_STORE_PATH)
        print("Placeholder Vector Store created.")


        global_llm = load_local_llm(model_name=LLM_MODEL_NAME)
        print("LLM loaded successfully.")

        global_retriever = get_retriever(global_vector_store)
        print("Retriever initialized globally.")
        
    except Exception as e:
        print(f"FATAL ERROR during model initialization: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Server startup failed: {e}")

    print("Server ready.")

@app.post("/index", tags=["Indexing"])
async def index_documents_endpoint(files: List[UploadFile] = File(...)):
    global global_retriever, global_vector_store
    
    if global_embed_model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG system is not yet initialized.")

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    indexed_count = 0
    DEFAULT_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        file_path = DEFAULT_DATA_FOLDER / file.filename
        
        # Files are saved regardless of type, and validation/loading happens in load_documents
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            indexed_count += 1
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            
    if indexed_count == 0:
        if len(list(DEFAULT_DATA_FOLDER.iterdir())) == 0:
            shutil.rmtree(DEFAULT_DATA_FOLDER)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files were successfully uploaded or saved.")


    print(f"Successfully saved {indexed_count} files to {DEFAULT_DATA_FOLDER}.")
    
    print("Re-indexing all documents and rebuilding vector store...")
    try:
        # load_documents now includes logic to prioritize supported file types
        all_docs = load_documents(DEFAULT_DATA_FOLDER)
        
        if not all_docs:
            if len(list(DEFAULT_DATA_FOLDER.iterdir())) == 0:
                 raise Exception("No documents found for indexing after upload.")
            
        chunks = chunk_documents(all_docs)
        
        # If chunks is empty (e.g., all documents were unreadable), use the dummy document
        if not chunks:
             chunks = [Document(page_content="No indexable content found. Knowledge base is empty.", metadata={"source": "Internal System"})]

        global_vector_store = build_vector_store(chunks, global_embed_model, DEFAULT_STORE_PATH)
        
        if global_llm is not None:
             global_retriever = get_retriever(global_vector_store)
             
        print(f"Indexing complete. {len(chunks)} chunks created and retriever updated.")
        
    except Exception as e:
        print(f"Critical error during re-indexing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Indexing failed internally: {e}. Check server logs.")

    return {"message": f"Indexed {indexed_count} uploaded files. Total chunks: {len(chunks)}", "num_chunks": len(chunks)}


@app.post("/query", response_model=QueryResponse, tags=["Querying"])
async def query_endpoint(request: QueryRequest):
    global global_retriever, global_llm, global_tokenizer
    
    if global_retriever is None or global_llm is None or global_tokenizer is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG system is not initialized.")
    
    try:
        docs = global_retriever.invoke(request.query)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = assemble_qwen_prompt(
            current_query=request.query, 
            context_text=context_text, 
            history=request.history, 
            system_prompt=RAG_SYSTEM_PROMPT, 
            tokenizer=global_tokenizer
        )
        
        output = global_llm.pipeline(prompt)
        
        answer = output[0]['generated_text'].strip()
        
        sources = set()
        for doc in docs:
            source_path = doc.metadata.get('source', 'Unknown')
            sources.add(Path(source_path).name) 
            
        return QueryResponse(
            answer=answer,
            sources=list(sources)
        )
        
    except Exception as e:
        print(f"Error during query processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred while processing the query.")
