import os
from typing import Protocol, Any, List

# --- Abstract Interfaces ---

class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...

class VectorStoreProvider(Protocol):
    def search(self, query: str) -> List[Any]:
        ...

class DatabaseProvider(Protocol):
    def execute(self, query: str) -> Any:
        ...

# --- Concrete Implementations: Local ---

class LocalOllamaProvider:
    def generate(self, prompt: str) -> str:
        return f"[Local Llama3] Generating response for: {prompt}"

class LocalChromaProvider:
    def search(self, query: str) -> List[Any]:
        return ["[Local Chroma] Result 1"]

class LocalDuckDBProvider:
    def execute(self, query: str) -> Any:
        print(f"[Local DuckDB] Executing: {query}")
        return []

# --- Concrete Implementations: Cloud ---

class CloudGeminiProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        return f"[Cloud Gemini] Generating response for: {prompt}"

class CloudPineconeProvider:
    def __init__(self, api_key: str, env: str):
        self.api_key = api_key
        self.env = env

    def search(self, query: str) -> List[Any]:
        return ["[Cloud Pinecone] Result 1"]

class CloudPostgresProvider:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def execute(self, query: str) -> Any:
        print(f"[Cloud RDS] Executing: {query}")
        return []

# --- Factory Pattern ---

class ServiceFactory:
    """
    Factory to swap between local and cloud providers based on the environment variables.
    This ensures Modular Integrity via Dependency Injection.
    """
    @staticmethod
    def get_llm_provider() -> LLMProvider:
        provider = os.getenv("LLM_PROVIDER", "local")
        if provider == "cloud":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY must be set for cloud provider failover.")
            return CloudGeminiProvider(api_key=api_key)
        return LocalOllamaProvider()

    @staticmethod
    def get_vector_store() -> VectorStoreProvider:
        provider = os.getenv("VECTOR_STORE_PROVIDER", "local")
        if provider == "cloud":
            api_key = os.getenv("PINECONE_API_KEY")
            env = os.getenv("PINECONE_ENV", "us-east-1")
            if not api_key:
                raise ValueError("PINECONE_API_KEY must be set for cloud vector store.")
            return CloudPineconeProvider(api_key=api_key, env=env)
        return LocalChromaProvider()

    @staticmethod
    def get_database() -> DatabaseProvider:
        provider = os.getenv("DB_PROVIDER", "local")
        if provider == "cloud":
            conn_str = os.getenv("POSTGRES_CONNECTION_STRING")
            if not conn_str:
                raise ValueError("POSTGRES_CONNECTION_STRING must be set for cloud DB.")
            return CloudPostgresProvider(connection_string=conn_str)
        return LocalDuckDBProvider()
