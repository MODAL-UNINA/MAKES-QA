# %%
from pathlib import Path
from langgraph.graph import StateGraph, END
from agents.construction_agents import (
    PreProcessingAgent,
    EntityExtractorAgent,
    RelationExtractorAgent,
    NormalizationAgent,
)
from agents.expansion_agent import KGExpansionAgent
import uuid
from concurrent.futures import ThreadPoolExecutor
from langgraph.types import Command
import pickle
import argparse
from utils.utils import load_model_config
from utils.utils import ModelConfig
from utils.classes import (
    KGExtractionState,
    MainKGExtractionState,
)

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument(
    "--config", type=str, default="configs/config_models.yaml", help="Path to model configuration file"
)
parser.add_argument(
    "--dir_path",
    type=str,
    default="data",
    help="Path to the directory containing PDF files.",
)

parser.add_argument(
    "--download_dir",
    type=str,
    default="data/new_papers",
    help="Directory to save downloaded papers.",
)

parser.add_argument(
    "--expansion_choice",
    type=str,
    default="1",
    help="Choice of expansion strategy: 1 for entity-based search, 2 for foundational papers, 3 for common citations.",
)

parser.add_argument(
    "--results_dir",
    type=str,
    default="results",
    help="Directory to save results.",
)

parser.add_argument(
    "--new_papers_number",
    type=int,
    default=3,
    help="Number of new papers to download for kg expansion.",
)

class KGExtractor:
    def __init__(self, model_config: ModelConfig):
        self.preprocessing_agent = PreProcessingAgent(
            model_config.get_model(PreProcessingAgent)
        )
        self.entity_extractor = EntityExtractorAgent(
            model_config.get_model(EntityExtractorAgent)
        )
        self.relation_extractor = RelationExtractorAgent(
            model_config.get_model(RelationExtractorAgent)
        )
        self.workflow = self._create_subgraph_workflow()

    def _create_subgraph_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(KGExtractionState)

        # Add nodes
        workflow.add_node("preprocess_paper", self._preprocess_paper)

        workflow.add_node("extract_parallel", self._extract_parallel)

        # Add edges
        workflow.set_entry_point("preprocess_paper")

        workflow.add_edge("preprocess_paper", "extract_parallel")
        workflow.add_edge("extract_parallel", END)

        return workflow.compile()

    def _preprocess_paper(self, state: KGExtractionState) -> KGExtractionState:
        """Preprocess the scientific paper text"""

        research_topics, chunks = self.preprocessing_agent(state)

        return Command(update={"research_topics": research_topics, "chunks": chunks})

    def _process_chunk(
        self, chunk: str, chunk_id: int, research_topics, paper_source: str
    ):
        """Process a single chunk: extract entities then relations"""
        # Extract entities
        entities = self.entity_extractor.extract_entities(
            chunk, chunk_id, research_topics
        )

        # Extract relations
        triples, entities_new = self.relation_extractor.extract_relations(
            chunk, entities, chunk_id, paper_source
        )

        return entities_new, triples

    def _extract_parallel(self, state: KGExtractionState) -> KGExtractionState:
        """Extract entities and relations in parallel"""

        chunk_entities = {}
        chunk_triples = {}
        all_triples = []
        all_entities = []

        for i, chunk in enumerate(state["chunks"]):
            entities, triples = self._process_chunk(
                chunk,
                i,
                research_topics=state["research_topics"],
                paper_source=state["paper_source"],
            )
            chunk_entities[i] = entities
            chunk_triples[i] = triples
            all_triples.extend(triples)
            all_entities.extend(entities)

        duplicated_entities = set()
        entities_deduplicated = []

        for entity in all_entities:
            key = (
                entity.name.strip(),
                entity.type.strip(),
            )
            if key not in duplicated_entities:
                duplicated_entities.add(key)
                entities_deduplicated.append(entity)

        duplicated_triples = set()
        triples_deduplicated = []

        for triple in all_triples:
            key = (triple.head.strip(), triple.relation.strip(), triple.tail.strip())
            if key not in duplicated_triples:
                duplicated_triples.add(key)
                triples_deduplicated.append(triple)

        return Command(
            update={
                "chunk_entities": chunk_entities,
                "chunk_relations": chunk_triples,
                "all_entities": entities_deduplicated,
                "all_triples": triples_deduplicated,
            }
        )


class KGAggregator:
    """Main orchestrator for the knowledge graph extraction workflow"""

    def __init__(self, model_config: ModelConfig):

        self.model_config = model_config
        self.normalization_agent = NormalizationAgent(
            model_config.get_model(NormalizationAgent)
        )
        self.expansion_agent = KGExpansionAgent(
            model_config.get_model(KGExpansionAgent)
        )
        self.main_workflow = self._create_main_workflow()

    def _create_main_workflow(self) -> StateGraph:
        """Create the main workflow"""

        main_workflow = StateGraph(MainKGExtractionState)

        main_workflow.add_node("init_state", self._init_state)
        main_workflow.add_node(
            "parallel_document_processing", self._parallel_document_processing
        )
        main_workflow.add_node("aggregate_kg", self._aggregate_kg)
        main_workflow.add_node("normalize_kg", self._normalize_kg)
        main_workflow.add_node("search_new_papers", self._search_new_papers)
        main_workflow.add_node("router", self._router)

        # Set entry point
        main_workflow.set_entry_point("init_state")
        main_workflow.add_edge("init_state", "parallel_document_processing")
        main_workflow.add_edge("parallel_document_processing", "aggregate_kg")
        main_workflow.add_edge("aggregate_kg", "normalize_kg")

        main_workflow.add_conditional_edges(
            "normalize_kg",
            self._router,
            {
                "continue": "search_new_papers",
                "END": END,
            }

        )

        main_workflow.add_conditional_edges(
            "search_new_papers",
            self._router,
            {
                "continue": "parallel_document_processing",
                "END": END,
            }
        )

        return main_workflow.compile()

    def _init_state(self, state: MainKGExtractionState) -> Command:
        """Initialize the state for the main workflow"""
        return Command(
            update={
                "counter": 0,
            }
        )

    def _router(self, state: MainKGExtractionState) -> str:
        """Route to the next step based on the state"""

        if state["counter"] == 1 and len(state["documents"]) > 0:
            return "continue"
        else:
            return "END"

    def _process_single_document(self, source: str) -> KGExtractionState:
        """Process a single source document to extract knowledge graph"""

        print(f"Processing source: {source}\n")
        extractor = KGExtractor(self.model_config)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = KGExtractionState(paper_source=source)

        # Execute the workflow
        result = extractor.workflow.invoke(initial_state, config)

        return result

    def _parallel_document_processing(
        self, state: MainKGExtractionState
    ) -> MainKGExtractionState:
        """Process multiple documents in parallel"""
        
        with ThreadPoolExecutor(
            max_workers=min(len(state["documents"]), 32)
        ) as executor:
            results = list(
                executor.map(self._process_single_document, state["documents"])
            )

        return Command(
            update={
                "document_results": results,
            }
        )

    def _aggregate_kg(self, state: MainKGExtractionState) -> MainKGExtractionState:
        """Aggregate entities and triples from all processed documents"""

        all_entities = []
        all_triples = []

        for doc_result in state["document_results"]:
            if doc_result["all_entities"]:
                all_entities.extend(doc_result["all_entities"])
            if doc_result["all_triples"]:
                all_triples.extend(doc_result["all_triples"])

        entity_keys = set()
        deduplicated_entities = []

        for entity in all_entities:
            key = (
                entity.name.strip(),
                entity.type.strip(),
            )
            if key not in entity_keys:
                entity_keys.add(key)
                deduplicated_entities.append(entity)

        triple_keys = set()
        deduplicated_triples = []

        for triple in all_triples:
            key = (triple.head.strip(), triple.relation.strip(), triple.tail.strip())
            if key not in triple_keys:
                triple_keys.add(key)
                deduplicated_triples.append(triple)

        if state["counter"] == 1:
            return Command(
                update={
                    "aggregated_entities_new": deduplicated_entities,
                    "aggregated_triples_new": deduplicated_triples,
                }
            )
        else:
            return Command(
                update={
                    "aggregated_entities": deduplicated_entities,
                    "aggregated_triples": deduplicated_triples,
                }
            )

    def _normalize_kg(self, state: MainKGExtractionState) -> Command:
        """Normalize the aggregated knowledge graph"""
        
        print(f"Entities: {len(state['aggregated_entities'])}\nTriples: {len(state['aggregated_triples'])}\n")
        print("Normalizing knowledge graph...\n")

        if state["counter"] == 0:
            normalized_entities, entity_canonical_mappings = (
                self.normalization_agent.normalize_entities(
                    state["aggregated_entities"]
                )
            )
            triples_standard = self.normalization_agent.standardize_relations(
                state["aggregated_triples"], entity_canonical_mappings
            )
            standardized_triplets = (
                self.normalization_agent.normalize_triples(
                    triples_standard, normalized_entities
                )
            )
        else:
            normalized_entities, entity_canonical_mappings = (
                self.normalization_agent.normalize_entities_lists(
                    state["aggregated_entities_new"], state["normalized_entities"]
                )
            )
            triples_standard_1 = self.normalization_agent.standardize_relations(
                state["normalized_triples"], entity_canonical_mappings
            )
            triples_standard_2 = self.normalization_agent.standardize_relations(
                state["aggregated_triples_new"], entity_canonical_mappings
            )
            standardized_triplets = (
                self.normalization_agent.normalize_triples_list(
                    triples_standard_1, triples_standard_2, normalized_entities
                )
            )

        counter = state["counter"] + 1
        print(f"Normalized entities: {len(normalized_entities)}\nNormalized triples: {len(standardized_triplets)}\n")
        return Command(
            update={
                "normalized_entities": normalized_entities,
                "normalized_triples": standardized_triplets,
                "counter": counter,
            }
        )
        

    def _search_new_papers(self, state: MainKGExtractionState) -> Command:
        """Search for new papers based on the extracted knowledge graph"""

        print("Searching for new papers based on the expansion choice...\n")

        # Use the expansion agent to find new papers
        new_papers = self.expansion_agent.search_new_papers(state)

        return Command(
            update={
                "documents": new_papers,
            }
        )

# %%

if __name__ == "__main__":

    args, _ = parser.parse_known_args()

    # Get all PDF files in the directory
    documents = [str(file) for file in Path(args.dir_path).glob("*.pdf")]

    model_config = load_model_config(args.config)

    aggregator = KGAggregator(model_config=model_config)

    initial_state = MainKGExtractionState(
        documents=documents, download_dir=args.download_dir, expansion_choice=args.expansion_choice, new_papers_number=args.new_papers_number
    )

    result = aggregator.main_workflow.invoke(initial_state)

    triples = result["normalized_triples"]

    with open(f"{args.results_dir}/triples.pkl", "wb") as f:
        pickle.dump(triples, f)
# %%
