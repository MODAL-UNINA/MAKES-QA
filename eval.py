# %%
from typing import List, Dict
import re
from langchain.schema import Document
import os
import pymupdf
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import networkx as nx
import json
from langchain_core.prompts import ChatPromptTemplate
import unicodedata
from pathlib import Path
import argparse
import yaml
from utils.utils import get_llm, load_knowledge_graph
from utils.classes import Triple, Score, Query, Score

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument(
    "--config",
    type=str,
    default="configs/config_qa.yaml",
    help="Path to the configuration file.",
)

parser.add_argument(
    "--dir_path",
    type=str,
    default="data",
    help="Path to the directory containing PDF files.",
)

parser.add_argument(
    "--results_dir",
    type=str,
    default="results",
    help="Directory to save results.",
)


class QueryGenerator:
    """
    Generates search queries from knowledge triples for retrieval.
    """

    def __init__(self, model_name: str):

        self.llm = get_llm(model_name)
        self.system_prompt = """You are a Query Generator. Your task is to convert a knowledge triple into a search query 
        that will help retrieve relevant passages from a scientific paper to evaluate if the triple is supported by the paper.
        
        The query should be concise but include key terms from the triple that will maximize the chance of finding relevant information.
        """

    def generate_query(self, triple: Triple) -> str:
        """
        Generates a search query from a knowledge triple.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at generating concise and effective search queries to retrieve relevant scientific information.

        You will be given a factual relationship in the form: "head -> relation -> tail", along with a short description that provides additional context.

        Your task is to generate a search query that could be used to find relevant content in a scientific paper about this relationship.

        Instructions:
        - Use only the most meaningful and distinctive terms from the relationship and the description.
        - The query must be concise (5–10 words).
        - Do NOT include any explanation, formatting, or extra text.
        - Return ONLY the search query string.
        """,
                ),
                (
                    "human",
                    """Relationship: {head} -> {relation} -> {tail}
                    Description: {description}""",
                ),
            ]
        )

        chain = prompt | self.llm.with_structured_output(Query)
        response = chain.invoke(
            {
                "head": triple.head,
                "relation": triple.relation,
                "tail": triple.tail,
                "description": triple.description if triple.description else "",
            }
        )

        return response.query


class DocumentRetriever:
    """
    Retrieve relevant passages from scientific documents using a vector store.
    """

    def __init__(
        self,
        documents,
        page_overlap: int = 200,
        embedding_model: str = "allenai/scibert_scivocab_uncased",
        model_name: str = "qwen2.5:14b",
    ):

        self.llm = get_llm(model_name)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = self.build_vector_store(documents, page_overlap)

    def build_vector_store(self, documents, page_overlap):
        """ 
        Creates a vector store from the provided documents.
        """

        split_docs = []

        for doc in documents:
            doc_text = doc["text"]
            prev_text_tail = ""
            for i, page in enumerate(doc_text):
                text = page.get_text("text")
                text = re.sub(r"[\ue000-\uf8ff]", "", text)
                text = unicodedata.normalize("NFKC", text)

                if i > 0:
                    text = prev_text_tail + "\n" + text

                prev_text_tail = text[-page_overlap:]

                split_docs.append(
                    Document(
                        page_content=text.strip().lower(), metadata={"chunk_id": i}
                    )
                )

        texts = [doc.page_content for doc in split_docs]
        metadatas = [doc.metadata for doc in split_docs]

        vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)

        return vector_store

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Searches for relevant passages in the vector store based on the query.
        """

        query_norm = query.strip().lower()  
        docs = self.vector_store.similarity_search(query_norm, k=top_k)

        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

    def evaluate_retrieval(self, query: str, retrieved_passages: List[str]) -> float:

        context = "\n\n".join(retrieved_passages)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert evaluator of information retrieval quality.

        Your task is to rate how relevant a set of retrieved text passages is to a given search query.
        You must return only a numeric score between 0 (not relevant at all) and 1 (perfectly relevant). /no_think
        """,
                ),
                (
                    "human",
                    """Search Query:
        \"\"\"{query}\"\"\"

        Retrieved Passages:
        \"\"\"{context}\"\"\"

        Please rate the relevance of the retrieved passages to the query.
        Only return the numeric score. Return ONLY a decimal number between 0.0 and 1.0. 
        Do not explain or justify your answer." /no_think""",
                ),
            ]
        )
        chain = prompt | self.llm.with_structured_output(Score)
        response = chain.invoke({"query": query, "context": context})

        return response.score

class KnowledgeGraphEvaluator:
    """
    Valuta un knowledge graph utilizzando il retrieval per ottenere contesto rilevante.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:14b",
        retriever: DocumentRetriever = None,
        top_k: int = 5
    ):
        
        self.llm = get_llm(model_name)
        self.retriever = retriever
        self.query_generator = QueryGenerator(
            model_name=model_name
        )
        self.top_k = top_k

    def evaluate_knowledge_graph(self, triples: List[Triple]) -> Dict[str, float]:
        """
        Evaluates a knowledge graph by analyzing its triples.
        """
        if not triples:
            return {
                "average_confidence": 0.0,
                "average_clarity": 0.0,
                "average_relevance": 0.0,
                "overall_quality": 0.0,
                "total_triples": 0,
            }

        total_confidence = 0.0
        total_clarity = 0.0
        total_relevance = 0.0

        for triple in triples:

            query = self.query_generator.generate_query(triple)

            relevant_passages = self.retriever.retrieve(
                query, top_k=self.top_k
            )
            context = "\n\n".join([p["content"] for p in relevant_passages])
            

            confidence = self._evaluate_confidence(triple, context)
            clarity = self._evaluate_clarity(triple)
            relevance = self._evaluate_relevance(triple, context)

            total_confidence += confidence
            total_clarity += clarity
            total_relevance += relevance

        num_triples = len(triples)
        avg_confidence = total_confidence / num_triples
        avg_clarity = total_clarity / num_triples
        avg_relevance = total_relevance / num_triples

        quality_score = (avg_confidence + avg_clarity + avg_relevance) / 3

        return {
            "average_confidence": round(avg_confidence, 3),
            "average_clarity": round(avg_clarity, 3),
            "average_relevance": round(avg_relevance, 3),
            "overall_quality": round(quality_score, 3),
            "total_triples": num_triples,
        }

    def _evaluate_confidence(self, triple: Triple, context: str) -> float:
        """
        Evaluates the confidence of a knowledge triple based on the context retrieved.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert in evaluating factual statements based on scientific evidence.

        You will be given:
        - Chunks of scientific text (the context).
        - A factual statement expressed as a triple: "head relation tail".

        Your task is to assess how well this statement is supported by the context.

        Guidelines:
        - Consider both **direct evidence** and **indirect inference** from the context.
        - The statement does not need to be explicitly stated verbatim — logical implication or paraphrased support is acceptable.
        - Focus on semantic alignment and whether the context gives sufficient confidence in the truth of the statement.

        Return ONLY a number between 0.0 (not supported at all) and 1.0 (strongly supported by context). Do not explain or justify your answer.
        /no_think
        """,
                ),
                (
                    "human",
                    """Context:
        \"\"\"{context}\"\"\"

        Statement:
        \"\"\"{head} {relation} {tail}\"\"\"""",
                ),
            ]
        )

        chain = prompt | self.llm.with_structured_output(Score)
        response = chain.invoke(
            {
                "context": context,
                "head": triple.head,
                "relation": triple.relation,
                "tail": triple.tail,
            }
        )

        score = response.score
        if not (0 <= score <= 1):
            return 0

        return score

    def _evaluate_clarity(self, triple: Triple) -> float:
        """
        Evaluates the clarity of a knowledge triple.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at evaluating the clarity and specificity of factual scientific relations.

        You will be given a factual statement in the form: "head relation tail".

        Evaluate the clarity and specificity of the relation.

        Consider:
        1. Are the entities precise and unambiguous?
        2. Is the type of relation specific and well-defined?
        3. Would experts interpret this uniformly?
        4. Are there vague terms that reduce clarity?

        Rate from 0.0 (very ambiguous) to 1.0 (perfectly clear).
        Return ONLY a decimal number between 0.0 and 1.0. /no_think""",
                ),
                (
                    "human",
                    """Statement:
        {head} {relation} {tail}""",
                ),
            ]
        )

        chain = prompt | self.llm.with_structured_output(Score)
        response = chain.invoke(
            {"head": triple.head, "relation": triple.relation, "tail": triple.tail}
        )

        score = response.score
        if not (0 <= score <= 1):
            return 0

        return score

    def _evaluate_relevance(self, triple: Triple, context: str) -> float:
        """
        Evaluates the relevance of a knowledge triple to the provided context.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at evaluating the relevance of factual scientific relations with respect to a given context.

        You will be given:
        - Chunks of context extracted from a scientific paper.
        - A factual statement in the form: "head relation tail".

        Evaluate how relevant the statement is to the provided content.

        Consider:
        1. Is it directly connected to the main themes of the context?
        2. Does it help understand the concepts, results, or methodologies described?
        3. Is it important for the scientific contribution discussed?
        4. Is it central to the topic addressed in the context?

        CRITICAL: You must respond with ONLY a single decimal number between 0.0 and 1.0.
        - 0.0 = completely irrelevant
        - 1.0 = perfectly relevant

        Do NOT include any explanation, reasoning, or additional text. /no_think
        """,
                ),
                (
                    "human",
                    """Context:
        {context}

        Statement:
        {head} {relation} {tail}""",
                ),
            ]
        )

        chain = prompt | self.llm.with_structured_output(Score)

        response = chain.invoke(
            {
                "context": context,
                "head": triple.head,
                "relation": triple.relation,
                "tail": triple.tail,
            }
        )

        score = response.score
        if not (0 <= score <= 1):
            return 0

        return score


class GraphStructureAnalyzer:
    """
    Analyzes the structure of a knowledge graph and computes various metrics.
    """

    def __init__(self, triples: List[Triple]):
        self.triples = triples
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for t in self.triples:
            G.add_edge(t.head, t.tail, relation=t.relation)
        return G

    def compute_metrics(self) -> Dict[str, float]:
        G = self.graph
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        undirected_G = G.to_undirected()

        average_degree = (
            sum(dict(G.degree()).values()) / float(num_nodes) if num_nodes > 0 else 0.0
        )
        avg_in_degree = (
            sum(dict(G.in_degree()).values()) / float(num_nodes)
            if num_nodes > 0
            else 0.0
        )
        avg_out_degree = (
            sum(dict(G.out_degree()).values()) / float(num_nodes)
            if num_nodes > 0
            else 0.0
        )

        try:
            avg_clustering = nx.average_clustering(undirected_G)
        except Exception:
            avg_clustering = 0.0

        try:
            if nx.is_connected(undirected_G):
                diameter = nx.diameter(undirected_G)
                avg_shortest_path_length = nx.average_shortest_path_length(undirected_G)
            else:
                diameter = -1
                avg_shortest_path_length = -1
        except Exception:
            diameter = -1
            avg_shortest_path_length = -1

        try:
            mst = nx.minimum_spanning_tree(undirected_G)
            redundancy_ratio = (
                (num_edges - mst.number_of_edges()) / num_edges
                if num_edges > 0
                else 0.0
            )
        except Exception:
            redundancy_ratio = 0.0

        metrics = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": nx.density(G),
            "average_clustering": avg_clustering,
            "average_degree": average_degree,
            "average_in_degree": avg_in_degree,
            "average_out_degree": avg_out_degree,
            "diameter": diameter,
            "avg_shortest_path_length": avg_shortest_path_length,
            "estimated_redundancy": redundancy_ratio,
            "is_weakly_connected": nx.is_weakly_connected(G),
            "is_strongly_connected": nx.is_strongly_connected(G),
            "num_weakly_connected_components": nx.number_weakly_connected_components(G),
            "num_strongly_connected_components": nx.number_strongly_connected_components(
                G
            ),
        }
        return metrics


def extract_text_from_file(file_path: str) -> str:
    """
    Load text from a PDF file.
    """
    if file_path.endswith(".pdf"):
        doc = pymupdf.open(file_path)

        return doc
    else:
        raise ValueError(
            f"Unsupported file format: {file_path}. Only PDF files are supported."
        )


def combine_papers_text(paper_paths: List[str]) -> str:
    """
    Combine the text from multiple papers into a single dictionary.
    """
    paper_texts = {}

    for paper_path in paper_paths:
        if not os.path.exists(paper_path):
            continue

        paper_text = extract_text_from_file(paper_path)

        paper_texts[paper_path] = paper_text

    return paper_texts


def evaluate_kg_with_rag(
    triples,
    paper_paths: List[str] = None,
    model_name="qwen2.5:14b",
    embedding_model="allenai/scibert_scivocab_uncased",
    top_k=5
):
    """
    Evaluate a knowledge graph using retrieval-augmented generation (RAG) with the provided papers.
    """

    paper_texts = combine_papers_text(paper_paths)

    documents = [{"text": text, "title": title} for title, text in paper_texts.items()]

    retriever = DocumentRetriever(
        documents, embedding_model=embedding_model, model_name=model_name
    )

    evaluator = KnowledgeGraphEvaluator(
        model_name=model_name, retriever=retriever, top_k=top_k
    )
    results = evaluator.evaluate_knowledge_graph(triples)

    analyzer = GraphStructureAnalyzer(triples)
    structure_metrics = analyzer.compute_metrics()
    results.update(structure_metrics)

    return results


# %%
if __name__ == "__main__":

    args, _ = parser.parse_known_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    triples = load_knowledge_graph(config["kg_path"])

    papers = [str(file) for file in Path(args.dir_path).glob("*.pdf")]

    try:
        results = evaluate_kg_with_rag(
            triples=triples,
            paper_paths=papers,
            model_name=config["model_name"],
            embedding_model=config["embedding_model"],
            top_k=config["top_k"]
        )
        
        eval_file_path = f"{args.results_dir}/eval.json"

        with open(eval_file_path, "w", encoding="utf-8") as json_file:
            json.dump(results, json_file, indent=4)

    except Exception as e:
        print(f"Error during evaluation: {e}")

# %%
