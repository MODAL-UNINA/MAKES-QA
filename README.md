# MAKES-QA: A Multi-Agent framework for Knowledge graph construction and Enrichment over Scientific literature for Question Answering

## Abstract
Knowledge Graphs (KGs) offer an effective framework for organizing complex information, yet their construction and use for question answering face significant challenges due to the dynamic and evolving nature of research domains. In this work, we present MAKES-QA, a modular and dynamic multi-agent framework for KG-based question answering (KGQA) that leverages Large Language Models (LLMs) to extract, normalize, and integrate knowledge, particularly in the context of scientific literature. The system enables iterative enrichment of the KG through user-guided strategies, facilitating the discovery and integration of new information. Once constructed, the KG supports natural language queries via a Retrieval-Augmented Generation (RAG) approach that combines semantic retrieval of relevant triples with LLM-based answer synthesis. This architecture ensures accurate, context-aware responses grounded in curated scientific knowledge. Our approach promotes efficient exploration of scientific domains, reducing the need for exhaustive manual reading while enabling flexible knowledge discovery.

![Framework](framework.png)

## Prerequisites
This project requires a local Ollama instance to run language models. All setup instructions are available here: [Docker Ollama](https://hub.docker.com/r/ollama/ollama).

## Requirements
To set up the environment, create a python environment using the provided kg_env.yaml file:
```bash
conda env create -f kg_env.yaml
conda activate kg_env
```

## Usage
### 1. Build and Expand the Knowledge Graph
Run the full pipeline using:
```bash
python main.py
```
### Key arguments
```bash
--config CONFIG_PATH              
--dir_path DATA_DIR              
--download_dir NEW_PAPER_DIR      
--expansion_choice [1|2|3]                        
--new_papers_number N            
--results_dir RESULTS_DIR       
```

The default CONFIG_PATH is `configs/config_models.yaml`, it specifies which language models are used by each agent in the KG construction pipeline.

```yaml
default_model: qwen2.5:14b  
overrides:
  EntityExtractorAgent: qwen2.5:32b      
  RelationExtractorAgent: qwen2.5:32b      
```

### 2. Question Answering
Once the KG is built and stored in the `results_dir` specified during the construction phase, you can query it using:

```bash
python qa.py --query "What are the main methods discussed?"
```

### Key arguments
```bash
--config CONFIG_PATH 
```

The default CONFIG_PATH is `configs/config_qa.yaml`, which defines parameters for the question answering pipeline.

```yaml
kg_path: "results/triples.pkl"
model_name: "qwen2.5:32b"
embedding_model: "allenai/scibert_scivocab_uncased"
top_k: 5       
```

### 3. Evaluation
To evaluate the quality of the constructed Knowledge Graph (KG), both structurally and semantically, run the following script:

```bash
python eval.py --config configs/config_qa.yaml --dir_path data --results_dir results
```
