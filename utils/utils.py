import yaml
from langchain_ollama import ChatOllama
import requests
import os
import pickle 

class ModelConfig:
    def __init__(self, default_model: str, overrides: dict = None):
        self.default_model = default_model
        self.overrides = overrides or {}

    def get_model(self, agent_class):
        return self.overrides.get(agent_class.__name__, self.default_model)


def load_model_config(config_path: str = "config.yaml") -> ModelConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return ModelConfig(
        default_model=config.get("default_model", "qwen2.5:14b"),
        overrides=config.get("overrides", {})
    )

def get_llm(model_name="qwen2.5:14b") -> ChatOllama:
    """Initialize the LLM with the specified model name."""
    
    return ChatOllama(
        model=model_name
    )
        
def load_existing_titles(download_dir):
    """
    Load existing titles from the download directory.
    """
    
    existing_titles = []
    
    for filename in os.listdir(download_dir):
        if filename.endswith(".pdf"):
            title = os.path.splitext(filename)[0]
            existing_titles.append(title)
            
    return existing_titles

def download_pdf(url: str, path: str):

    response = requests.get(url)
    response.raise_for_status()

    if response.headers['content-type'] != 'application/pdf':
        raise Exception('The response is not a pdf')

    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def load_knowledge_graph(kg_path: str):
    """Loads the knowledge graph from a file."""

    with open(kg_path, 'rb') as f:
        kg_data = pickle.load(f)
    
    if 'normalized_triples' in kg_data:
        triples = kg_data['normalized_triples']
    elif isinstance(kg_data, list):
        triples = kg_data
    else:
        raise ValueError("Unknown knowledge graph format")
        
    return triples