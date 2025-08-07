#%%
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate 
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from collections import defaultdict
import argparse
import yaml
from utils.classes import Triple, Answer
from utils.utils import get_llm, load_knowledge_graph

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument(
    "--config",
    type=str,
    default="configs/config_qa.yaml",
    help="Path to the configuration file."
)

parser.add_argument(
    "--query",
    type=str,
    default="What are the main methods discussed?",
    help="Query to search in the knowledge graph."
)

class KGVectorStore:
    """Knowledge Graph Vector Store for Question Answering"""
    
    def __init__(self, triples: List[Triple], config):
        
        self.triples = triples
        self.llm = get_llm(config['model_name'])
        self._build_triple_documents()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config['embedding_model'],
            encode_kwargs = {'normalize_embeddings': True},
            query_encode_kwargs = {'normalize_embeddings': True}
        )        
        self._build_vector_store()
        
    def _build_triple_documents(self):
        """Builds documents from triples for vector store indexing."""
        self.triple_documents = []
        
        for i, triple in enumerate(self.triples):
            triple_text = f"{triple.head} {triple.relation} {triple.tail}."
            if triple.description:
                triple_text += f" {triple.description}"
            
            normalized_text = triple_text.strip().lower()
            
            doc = Document(
                page_content=normalized_text,
                metadata={
                    "triple_id": i,
                    "paper_id": triple.paper_id,
                }
            )
            self.triple_documents.append(doc)

    def _build_vector_store(self):

        texts = [doc.page_content for doc in self.triple_documents]
        metadatas = [doc.metadata for doc in self.triple_documents]
        
        self.vector_store = FAISS.from_texts(
            texts, 
            self.embeddings, 
            metadatas=metadatas
        )
 
    def search_relevant_triples(self, 
                              query: str, 
                              top_k: int = 5
                              ):
        """
        Searches for relevant triples in the knowledge graph based on the query.
        """
        query_norm = query.strip().lower()
        
        docs = self.vector_store.similarity_search_with_score(
                query_norm, k=top_k
            )
            
        retrieved_triples = []
        
        for doc, _ in docs:
                triple_id = doc.metadata["triple_id"]
                triple_obj = self.triples[triple_id]
                retrieved_triples.append(triple_obj)
            
        return retrieved_triples

class QA_Agent:
    """Agent for answering questions using a knowledge graph and RAG (Retrieval-Augmented Generation) approach."""
    
    def __init__(self, query, knowledge_graph, config):
        self.kg = knowledge_graph
        self.llm = get_llm(config['model_name'])
        self.query = query
        self.top_k = config['top_k']
        
    def answer_question(self) -> Dict[str, Any]:
        """Generates an answer to the question using the knowledge graph."""
        
        relevant_triples = self.kg.search_relevant_triples(
            query=self.query, 
            top_k=self.top_k, 
        )
        
        if not relevant_triples:
            return "I couldn't find relevant information in the knowledge graph to answer this question."

        context = self._build_context(relevant_triples)
        
        system_prompt = """You are an expert AI assistant specializing in responding to scientific question.
        
        You will be provided with factual information extracted from a knowledge graph and a specific research question.
        
        Your task is to reason over the context, identify relevant connections, and construct a precise and well-reasoned answer based solely on the provided information.
        
        Guidelines:
        - Respond in a single, complete sentence
        - Use formal academic language
        - Do not include explanations
        - Do not use external knowledge beyond the provided context
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """CONTEXT from the knowledge graph:
        {context}

        QUESTION: {question}

        ANSWER: """)
        ])
        
        chain = prompt | self.llm.with_structured_output(Answer)
        response = chain.invoke({"question": self.query, "context": context})
        
        return response.answer
    
    def _build_context(self, relevant_triples: List[Triple]) -> str:
        
        relation_groups = defaultdict(list)
        for triple in relevant_triples:
            relation_groups[triple.relation].append(triple)
        
        context_parts = []
        
        for relation, triples in relation_groups.items():
            context_parts.append(f"\n{relation.upper()}:")
            for triple in triples:
                description = triple.description if triple.description else ""
                context_parts.append(f"- {triple.head} {relation} {triple.tail}. {description}")
        
        return "\n".join(context_parts)


# %%
if __name__ == "__main__":
    
    args, _ = parser.parse_known_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    kg = load_knowledge_graph(config['kg_path'])
    
    vector_store = KGVectorStore(kg, config)

    agent = QA_Agent(query=args.query, knowledge_graph=vector_store, config=config)
    answer = agent.answer_question()

    print(f"QUESTION: {args.query}")
    print(f"\nANSWER: {answer}")

