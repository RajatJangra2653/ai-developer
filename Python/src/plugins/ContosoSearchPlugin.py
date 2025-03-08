import json
import os
from typing import Dict, List, Any, Optional

import requests
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

class ContosoSearchPlugin:
    """Plugin for semantic search of the Contoso Handbook using text embeddings."""
    
    def __init__(self):
        """Initialize the ContosoSearchPlugin with configuration from environment variables."""
        load_dotenv()
        
        # Azure OpenAI settings for embeddings
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.embedding_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        
        # Azure AI Search settings
        self.search_endpoint = os.getenv("AI_SEARCH_URL")
        self.search_key = os.getenv("AI_SEARCH_KEY")
        self.search_index_name = os.getenv("AZURE_SEARCH_INDEX", "employeehandbook")
        
        # Create search client
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.search_index_name,
            credential=AzureKeyCredential(self.search_key)
        )
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for the input text using Azure OpenAI."""
        if not text:
            raise ValueError("Input text cannot be empty")
            
        url = f"{self.openai_endpoint}/openai/deployments/{self.embedding_deployment}/embeddings?api-version={self.embedding_api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.openai_api_key
        }
        payload = {
            "input": text,
            "dimensions": 1536  # Standard for text-embedding-ada-002
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            embedding_data = response.json()
            return embedding_data["data"][0]["embedding"]
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    def search_documents(self, query: str, top: int = 3) -> List[Dict[str, Any]]:
        """Search for documents using vector search with the query embedding."""
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            
            # Create a vectorized query
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top,
                fields="contentVector"
            )
            
            # Execute the search
            results = self.search_client.search(
                search_text=query,  # Also include text search for hybrid retrieval
                vector_queries=[vector_query],
                select=["id", "content", "page_num", "chunk_id"],
                top=top
            )
            
            # Format the results
            search_results = []
            for result in results:
                search_results.append({
                    "id": result["id"],
                    "content": result["content"],
                    "page_num": result.get("page_num", "Unknown"),
                    "chunk_id": result.get("chunk_id", "Unknown"),
                    "score": result["@search.score"]
                })
            
            return search_results
            
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")
    
    def query_handbook(self, query: str, top: int = 3) -> str:
        """Main method to query the Contoso Handbook with a user query."""
        try:
            results = self.search_documents(query, top)
            
            # Format the results into a nice response
            if not results:
                return "No relevant information found in the Contoso Handbook."
            
            response = f"Here's what I found in the Contoso Handbook about '{query}':\n\n"
            for i, result in enumerate(results, 1):
                response += f"Result {i} (Page {result['page_num']}):\n{result['content']}\n\n"
            
            return response
            
        except Exception as e:
            return f"Error querying the Contoso Handbook: {str(e)}"


# Example of how to use the plugin
if __name__ == "__main__":
    search_plugin = ContosoSearchPlugin()
    query = "What is Contoso's vacation policy?"
    result = search_plugin.query_handbook(query)
    print(result)