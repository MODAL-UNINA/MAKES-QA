from typing import List, Dict, Optional, TypedDict
from pydantic import BaseModel, Field
from dataclasses import field


class Entity(BaseModel):
    name: str = Field(description="Name of the entity.")
    type: str = Field(description="Type of the entity (e.g., 'Author', 'Paper', 'Method').")
    description: str = Field(description="Brief description of the entity, explaining its role or significance.")

class EntityList(BaseModel):
    entities: List[Entity] = Field(
        description="List of entities extracted from the text chunk, each with a name, type, and description."
    )

class Triple(BaseModel):
    head: str = Field(description="The subject of the triple.")
    relation: str = Field(description="The type of relationship between the head and tail entities.")
    tail: str = Field(description="The object of the triple.")
    description: str = Field(
        description="A brief explanation of the relationship, providing context or meaning."
    )
    paper_id: Optional[str] = Field(default=None, description="ID of the paper from which the triple was extracted.")

class TripleList(BaseModel):
    triples: List[Triple] = Field(
        description="List of triples extracted from the text chunk, each with a head, relation, tail, and description."
    )

class isSimilar(BaseModel):
    are_similar: bool = Field(
        description="True if the two objects express the same meaning or concept."
    )

class ResearchTopics(BaseModel):
    topics: List[str] = Field(
        description="List of research topics extracted from the scientific paper."
    )

class Summary(BaseModel):
    summary: str = Field(
        description="Concise summary of the scientific paper, focusing on the main contributions and findings."
    )

class PaperMetadata(BaseModel):
    title: str = Field(description="Title of the paper.")
    authors: List[str] = Field(description="List of authors of the paper.")
    doi: str = Field(default=None, description="DOI of the paper.")
    

class KGExtractionState(TypedDict):
    paper_source: str = ""
    research_topics: List[str] = field(default_factory=list)
    chunks: List[str] = field(default_factory=list)
    chunk_entities: Dict[int, List[str]] = field(default_factory=dict)
    chunk_relations: Dict[int, List[Triple]] = field(default_factory=dict)
    all_triples: List[Triple] = field(default_factory=list)
    all_entities: List[Entity] = field(default_factory=list)


class MainKGExtractionState(TypedDict):
    documents: List[str] = field(default_factory=list)  
    document_results: List[KGExtractionState] = field(default_factory=list) 
    aggregated_entities: List[Entity] = field(default_factory=list) 
    aggregated_triples: List[Triple] = field(default_factory=list)
    aggregated_entities_new: List[Entity] = field(default_factory=list)  
    aggregated_triples_new: List[Triple] = field(default_factory=list)  
    normalized_entities: List[Entity] = field(default_factory=list)  
    normalized_triples: List[Triple] = field(default_factory=list) 
    download_dir: str = "" 
    counter: int = 0  
    expansion_choice: int = 0  
    new_papers_number: int = 3  

class Score(BaseModel):
    score: float = Field(description="The score of the evaluation, between 0.0 and 1.0.")

class Answer(BaseModel):
    answer: str = Field(description="The answer to the question.")

class Query(BaseModel):
    query: str = Field(
        description="The search query generated from the triple to retrieve relevant information."
    )

