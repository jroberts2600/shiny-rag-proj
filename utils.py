import os
import csv
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from unstructured.partition.image import partition_image
from unstructured.partition.pptx import partition_pptx
import requests
import re
from urllib3.exceptions import NewConnectionError
from requests.exceptions import RequestException, ConnectionError
import logging
import warnings

# Suppress the specific deprecation warning for HuggingFaceHub
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community.llms.huggingface_hub")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable parallelism for Hugging Face tokenizers to avoid potential deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Sentence Transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

def is_ollama_available(host="localhost", port="11434"):
    try:
        response = requests.get(f"http://{host}:{port}/api/tags", timeout=2)
        return response.status_code == 200
    except (requests.RequestException, ConnectionError, NewConnectionError):
        return False

def get_ollama_models():
    if is_ollama_available():
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                sorted_models = sorted(models, key=lambda x: get_param_count(x['name']))
                return [model['name'] for model in sorted_models]
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
    return []

def get_param_count(model_name):
    # Extract parameter count from model name (e.g., "llama2:7b" -> 7000000000)
    match = re.search(r':(\d+)b', model_name.lower())
    if match:
        return int(match.group(1)) * 1_000_000_000
    return float('inf')  # Put models without clear parameter count at the end

def verify_api_key_sync(api_key):
    if api_key.startswith("sk-") and len(api_key) == 51:
        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            if response.status_code == 200:
                return "OpenAI API key verified successfully! ✅", "GPT-3.5/4", "openai"
            else:
                return f"Invalid OpenAI API key ❌ (Status code: {response.status_code})", "Unknown", "unknown"
        except RequestException as e:
            return f"Error verifying OpenAI API key: {str(e)}", "Unknown", "unknown"
    elif api_key.startswith("sk-ant-"):
        try:
            # Use 'claude-2.1' as the default model for verification
            chat = ChatAnthropic(anthropic_api_key=api_key, model="claude-2.1")
            # We'll use a simple prompt to test the API key
            response = chat.invoke("Hello, this is a test.")
            if response:
                return "Claude API key verified successfully! ✅", "Claude", "claude"
            else:
                return "Invalid Claude API key ❌", "Unknown", "unknown"
        except Exception as e:
            return f"Error verifying Claude API key: {str(e)}", "Unknown", "unknown"
    elif api_key.startswith("hf_"):
        try:
            response = requests.get(
                "https://huggingface.co/api/models",
                headers={"Authorization": f"Bearer {api_key}"},
                params={"sort": "lastModified", "limit": 1}
            )
            if response.status_code == 200:
                return "HuggingFace API key verified successfully! ✅", "HuggingFace", "huggingface"
            elif response.status_code == 401:
                return "Invalid HuggingFace API key ❌", "Unknown", "unknown"
            else:
                return f"Error verifying HuggingFace API key (Status code: {response.status_code})", "Unknown", "unknown"
        except RequestException as e:
            return f"Error verifying HuggingFace API key: {str(e)}", "Unknown", "unknown"
    else:
        return "Invalid API key format ❌", "Unknown", "unknown"

def process_csv(file_content):
    csv_data = StringIO(file_content.decode('utf-8'))
    reader = csv.reader(csv_data)
    return "\n".join([",".join(row) for row in reader])

def process_documents(files, api_key=None):
    processed_docs = []
    processed_files = []
    loaders = {
        '.docx': partition_docx,
        '.pdf': partition_pdf,
        '.html': partition_html,
        '.htm': partition_html,
        '.txt': partition_text,
        '.png': partition_image,
        '.jpeg': partition_image,
        '.jpg': partition_image,
        '.pptx': partition_pptx,
        '.csv': process_csv,
    }

    def load_file(file):
        try:
            if file.getbuffer().nbytes > MAX_FILE_SIZE:
                raise ValueError(f"File {file.name} exceeds the maximum size limit of {MAX_FILE_SIZE / (1024 * 1024)} MB")

            file_path = os.path.join("documents", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            _, file_extension = os.path.splitext(file.name)
            loader_function = loaders.get(file_extension.lower())
            if not loader_function:
                raise ValueError(f"Unsupported file type: {file_extension}")

            logger.info(f"Loading {file.name} with extension {file_extension}")
            if file_extension.lower() == '.csv':
                with open(file_path, 'rb') as f:
                    text = loader_function(f.read())
            else:
                elements = loader_function(file_path)
                text = "\n".join([element.text for element in elements if element.text])
            logger.info(f"Processed file {file.name} successfully.")
            return text, file.name
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {str(e)}")
            raise e

    max_workers = min(32, (os.cpu_count() or 1) + 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(load_file, files)
        for result, file_name in results:
            processed_docs.append(result)
            processed_files.append(file_name)

    # Split the documents into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = []
    for doc in processed_docs:
        texts.extend(text_splitter.split_text(doc))

    # Create embeddings and store them in Chroma
    if api_key:
        status, _, api_type = verify_api_key_sync(api_key)
        if api_type == "openai":
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        elif api_type in ["huggingface", "claude"]:
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        else:
            raise ValueError("Invalid API key")
    else:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    db = Chroma.from_texts(texts, embeddings)

    return db, processed_files

def get_huggingface_model(api_key=None):
    models = [
        "microsoft/Phi-3-mini-4k-instruct",
        "google/flan-t5-base",
        "google/flan-t5-small"
    ]
    
    for model in models:
        try:
            if api_key:
                return HuggingFaceHub(repo_id=model, huggingfacehub_api_token=api_key, model_kwargs={"temperature": 0.5, "max_length": 512})
            else:
                huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
                if not huggingfacehub_api_token:
                    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")
                return HuggingFaceHub(repo_id=model, model_kwargs={"temperature": 0.5, "max_length": 512})
        except Exception as e:
            logger.warning(f"Failed to initialize {model}: {str(e)}. Trying next model.")
    
    raise ValueError("Failed to initialize any HuggingFace model")

def get_model(model_name, api_key=None):
    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        return ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0)
    elif model_name in ["claude-3-5-sonnet-20240620", "claude-2.1"]:
        return ChatAnthropic(anthropic_api_key=api_key, model=model_name, temperature=0)
    elif model_name in ["Phi-3-mini-4k-instruct", "flan-t5-base", "flan-t5-small"]:
        return get_huggingface_model(api_key)
    elif is_ollama_available() and model_name in get_ollama_models():
        return Ollama(model=model_name)
    else:
        raise ValueError(f"Model {model_name} is not available or not properly configured. Please check your settings and try again.")

def query_model(db, query, api_key, model_name):
    try:
        chat = get_model(model_name, api_key)
        logger.info(f"Using model: {model_name}")

        qa = ConversationalRetrievalChain.from_llm(
            llm=chat,
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
        )
        
        result = qa.invoke({"question": query, "chat_history": []})
        return result["answer"]
    except ValueError as e:
        logger.error(f"Model configuration error: {str(e)}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Unexpected error: {str(e)}"

def get_all_models():
    available_models = {
        "OpenAI": ["gpt-3.5-turbo", "gpt-4"],
        "Claude": [
            "claude-3-5-sonnet-20240620",  # This is Claude 3.5 Sonnet
            "claude-2.1"
        ],
        "HuggingFace": ["Phi-3-mini-4k-instruct", "flan-t5-base", "flan-t5-small"],
    }
    if is_ollama_available():
        available_models["Ollama"] = get_ollama_models()
    return available_models

# Ensure the 'documents' directory exists
if not os.path.exists("documents"):
    os.makedirs("documents")