import os
from langfuse import get_client
from langchain_mariadb import MariaDBStore
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

# 1. Load a pretrained Sentence Transformer model
model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Langfuse SDK
# Replace with your actual credentials and host
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" 

langfuse = get_client()
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Check your credentials.")

# connection string
url = f"mariadb+mariadbconnector://{os.getenv('MARIADB_USER')}:{os.getenv('MARIADB_PASSWORD')}@localhost/langchain"

# Initialize vector store
vectorstore = MariaDBStore(
    embeddings=model,
    embedding_length=384,
    datasource=url,
    collection_name="my_docs",
)

# Example application logic with Langfuse tracing
with langfuse.start_as_current_span(name="mariadb-trace") as span:
    # Add documents to the MariaDB vector store
    vectorstore.add_documents(
        [
            Document(page_content="The sun is a star."),
            Document(page_content="The moon is a natural satellite.")
        ]
    )

    # Perform a similarity search
    results = vectorstore.similarity_search("Tell me about celestial bodies.")

    # Log the search result to the trace
    span.update_trace(
        # "search_result",
        # value=str(results),
        metadata={"query": "Tell me about celestial bodies."}
    )

    print(f"Search results: {results}")

langfuse.flush()
