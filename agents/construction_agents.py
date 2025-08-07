import os
from typing import List, Dict, Tuple
import re
import pymupdf
import unicodedata
from langchain_core.prompts import ChatPromptTemplate 
from utils.utils import get_llm
from utils.classes import Entity, EntityList, Triple, TripleList, ResearchTopics, Summary, KGExtractionState, isSimilar


class PreProcessingAgent:
    """Agent responsible for ingesting scientific papers and preparing them for processing"""
    
    def __init__(self, model_name: str = "qwen2.5:14b"):
        self.llm = get_llm(model_name=model_name)

    def __call__(self, state: KGExtractionState) -> KGExtractionState:

        first_page, paper_text = self.load_paper(state)
        research_topics = self.extract_research_topics(first_page)
        chunks = self.paper_summary(paper_text, research_topics)

        return research_topics, chunks
    
    def load_paper(self, state: KGExtractionState) -> KGExtractionState:
        """Load the scientific paper text from a file"""
        if not os.path.exists(state['paper_source']):
            raise FileNotFoundError(f"Paper file not found: {state['paper_source']}")
        
        if state['paper_source'].endswith('.pdf'):
            doc = pymupdf.open(state['paper_source'])
            paper_text = ""
            for page in doc:
                paper_text += page.get_text("text") + "\n"

            first_page = doc[0].get_text()
        
        return first_page, doc
    
    def extract_research_topics(self, first_page) :
        """Extract research topics from the scientific paper text"""

        system_prompt = """You are a highly skilled expert in scientific papers preprocessing.
        Your primary task is to analyze provided scientific articles and accurately extract the main research topics discussed in the text.
        
        Instructions:
            - Be as **precise and technical** as possible. Avoid generic terms like "Science" or "Engineering".
            - If the paper involves **multiple research areas**, include all relevant topics (e.g., "Medicine, Neurology, Stroke Diagnosis, Computer Science, Deep Learning, Medical Imaging").
            - Prefer domain-specific subfields over broad disciplines.

        Your output must be a clean, concise list of research topics. Do not include explanations, summaries, or any additional text—only the list of topics. 
        """

        prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_prompt,
                    ),
                    (
                        "human",
                        """Extract the research topics from this text:\n\n{first_page} """
                    ),
                ]
            )
        chain = prompt | self.llm.with_structured_output(ResearchTopics)
        response = chain.invoke({"first_page": first_page})
    
        return response.topics

    def paper_summary(self, doc, research_topics) :
        """Generate summaries of the scientific paper pages, focusing on the specified research topics."""

        overlap = 200

        chunks = []
        prev_text_tail = ""
        for i, page in enumerate(doc):
            text = page.get_text("text")
            text = re.sub(r'[\ue000-\uf8ff]', '', text)
            text = unicodedata.normalize('NFKC', text)

            if i > 0:
                text = prev_text_tail + "\n" + text

            chunks.append(text)
            prev_text_tail = text[-overlap:]

        system_prompt = """You are an expert in scientific text summarization.

        Your task is to generate an exhaustive and informative summary of a scientific paper page. 
        Make sure to highlight content relevant to the specified research topics.
        Don't skip important details.

        Provide only the summary with no additional text or formatting. 
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                (
                    "human",
                    """Given the research topics: {research_topics}, summarize the following scientific text:\n\n{chunk}"""
                ),
            ]
        )
        chain = prompt | self.llm.with_structured_output(Summary)
        summarized_chunks = []
        for chunk in chunks[:1]:
            response = chain.invoke({"chunk": chunk, "research_topics": research_topics})
            summarized_chunks.append(response.summary)

        return summarized_chunks
    

class EntityExtractorAgent:
    """Agent responsible for extracting entities from text chunks"""
    
    def __init__(self, model_name: str = "qwen2.5:14b"):
        self.llm = get_llm(model_name=model_name)

        self.system_prompt = """
            You are a domain-aware information extraction agent specialized in identifying scientific words ("entities") suitable for relationship extraction in the construction of a knowledge graph.

            You will be provided with:
            - A **text chunk** from a scientific document.
            - A list of **research topics** that define the thematic focus of the document.

            ## Your task:
            Extract a **concise** list of the **most salient** scientific entities from the text. Each entity must:
            1. Be **highly relevant** to the provided research topics.
            2. Represent a **central scientific concept** likely to participate in meaningful semantic relationships.
            3. Be expressed as a noun phrase (maximum 3 words).
            4. Be accompanied by:
            - A specific **entity type** (scientific role/category).
            - A brief **definition or contextual description** inferred from the text.

            ## Guidelines:
            - Focus only on entities that are **core to the scientific contributions or novel findings** of the document.
            - **Do not include**:
                - Generic terms, marginal concepts, or background knowledge not essential to the main research focus.
                - Decorative or isolated adjectives/nouns with no relational or conceptual value.
                - Author names, paper titles, citations, affiliations, figure/table mentions, or bibliographic data.
                - Redundant mentions (e.g., acronyms when a full term is present).
            - If in doubt, **exclude** the entity.
            - Prefer **precise, domain-specific** entity types over generic ones. You may extend the list if needed.

            ## Suggested Entity Types (extendable):
            - Model
            - Method or Procedure
            - Dataset or Corpus
            - Tool or Software
            - Task or Research Goal
            - Metric or Evaluation Criterion
            - Theory or Framework

            If no clearly relevant entities are found, return an empty list.

            Do not include explanations or commentary. 
            """

    def extract_entities(self, chunk: str, chunk_id: int, research_topics) -> List[str]:
        """Extract entities from a text chunk"""
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.system_prompt,
                    ),
                    (
                        "human",
                        """Extract entities from this text chunk, within the {research_topics} research topics. :\n\n{chunk}"""
                    ),
                ]
            )

            chain = prompt | self.llm.with_structured_output(EntityList)
            
            response = chain.invoke({"research_topics": research_topics, "chunk": chunk})
            entities = response.entities
                        
            return entities

            
        except Exception as e:
            print(f"Error extracting entities from chunk {chunk_id}: {e}")
            return []

class RelationExtractorAgent:
    """Agent responsible for extracting relations between entities"""
    
    def __init__(self, model_name: str = "qwen2.5:14b"):
        self.llm = get_llm(model_name=model_name)
        
        self.system_prompt = """
            You are an expert in extracting relationships from scientific texts for the construction of a knowledge graph.

            ## Task:
            Extract a **concise list of valid relationships** **only between the provided words** from a scientific text chunk.

            ## Inputs:
            - A list of words in the format: "Name [Type]: Description"
            - A scientific text chunk

            ## Output (for each relationship):
            - "head": subject word
            - "relation": a short verb phrase (max 3 words) that clearly expresses how the head relates to the tail
            - "tail": object word
            - "description": concise explanation from the text

            ## Extraction Guidelines:
            - Use only words from the list (no invented or inferred terms).
            - The triplet **head relation tail** must form a **grammatically correct and semantically meaningful sentence fragment** (as if part of a scientific paper).
                - Example: `"BERT" improves "NER"` is valid
                - Example: `"Attention" used "Transformer"` is invalid (incorrect syntax or inversion)
                - **Test each relation mentally as a phrase** before accepting it.
            - Extract each relationship **only once**, avoiding reverse duplicates. For example, if "A uses B" is extracted, **do not also extract "B is used by A"**, unless the reverse conveys a **clearly distinct meaning**.
            - Relations should reflect the actual **semantic directionality** expressed in the text.
            - Ignore irrelevant or decorative text.
            - If in doubt about clarity or relevance, **do not extract the relation**.
            - Return an empty list if no valid relations exist.

            ## Focus Areas:
            Prioritize relationships that express:
            - The **main research question** and what it addresses
            - The **methods** or approaches used to tackle a problem
            - The **references or prior works** that a method or idea is based on
            - The **key findings** or conclusions reached
            - The **core concepts** discussed and how they interact

            Respond with **only the extracted relationships**. 
            """
        
    def extract_relations(self, chunk: str, entities: List[str], chunk_id: int, paper_source) -> List[Triple]:
        """Extract relations from a text chunk given its entities"""
        
        if len(entities) < 2:
            return [], []  # Not enough entities to extract relations
        
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.system_prompt,
                    ),
                    ("human", 
                        """Extract relations from this text chunk.

                        Use only the following words:

                        {entities}

                        Text chunk:
                        {chunk}

                        Extract relations:""")

                ]
            )
            entity_block = "\n".join(
                f"- {e.name} [{e.type}]: {e.description}" for e in entities
            )

            chain = prompt | self.llm.with_structured_output(TripleList)
            response = chain.invoke({"entities": entity_block, "chunk": chunk})
            triples_raw = response.triples

            entity_names = {e.name for e in entities}
            valid_triples = [
                triple for triple in triples_raw
                if triple.head in entity_names and triple.tail in entity_names
            ]

            used_entities = {triple.head for triple in valid_triples} | {triple.tail for triple in valid_triples}
            entities = [e for e in entities if e.name in used_entities]

            for triple in valid_triples:
                triple.paper_id = paper_source

            return valid_triples, entities

        except Exception as e:
            print(f"Error extracting relations from chunk {chunk_id}: {e}")
            return [], []


class NormalizationAgent:
    """
    Agent responsible for normalizing entities and relations extracted from text chunks.
    """
    
    def __init__(self, model_name: str = "qwen2.5:14b"):
        """
        Initialize the agent with the specified model name.
        """
        self.llm = get_llm(model_name=model_name)
        

    def normalize_entities(self, entities: List[Dict[str, str]]) -> Dict[str, str]:
    
        purged_entities_preprompt = f"""
        You are an expert in semantic comparison of scientific and technical terminology.

        **TASK**:
        Determine whether two terms refer to the same scientific concept based on their meaning and function.

        **INPUT**:
        You will be given two terms, each with a short description.

        **MATCHING CRITERIA**:
        Return **"True"** only if all the following hold:
        - The two terms describe the **same underlying concept or object**, regardless of surface differences (e.g., abbreviation vs. full form, synonyms, singular/plural).
        - They refer to the **same level of abstraction** and **serve the same function** in scientific discourse.
        - They are **interchangeable** in context without loss of meaning.

        **DO NOT MATCH IF**:
        - One term is a subtype, variation, or implementation of the other.
        - The terms differ in **scope**, **purpose**, or **granularity**.
        - They describe different entities, even if related.

        **POSITIVE EXAMPLES (return "True")**:
        - "BERT" and "BERT model"
        - "Natural Language Processing" and "NLP"
        - "Large language model" and "Large language models"
        - "Convolutional Neural Network" and "CNN"

        **NEGATIVE EXAMPLES (return "False")**:
        - "Transformer" and "BERT" (BERT is a specific kind of Transformer)
        - "Language model" and "Large language model" (different specificity)
        - "Pretrained model" and "BERT" (BERT is an instance of the former)
        - "Word2Vec" and "GloVe" (different methods)

        **OUTPUT**:
        Respond with **only** "True" or "False". No explanations. 
        """

        purged_entities_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    purged_entities_preprompt,
                ),
                (
                    "human",
                    """Here are the two terms and their descriptions:

                    Term 1: {term1}
                    Description 1: {description1}

                    Term 2: {term2}
                    Description 2: {description2}
                            """),
                        ]
        )

        purged_entities_chain = (
            purged_entities_prompt
            | self.llm.with_structured_output(isSimilar)
        )

        normalized_entities_preprompt = """
        You are an expert in entity normalization for scientific documents.

        **TASK**:
        You will receive a list of 'Entity' objects, each defined as:
        - "name": the entity label as found in the text
        - "type": the category of the entity
        - "description": a brief explanation of what the entity refers to

        Your goal is to produce a **single normalized entity** that best represents the entire group.

        **INSTRUCTIONS**:
        - Extract and return **one representative entity** with:
            - A normalized **name** that is clear, specific, and general enough to cover all variations.
            - A **type** that accurately reflects the role or category of the entity across all inputs.
            - A **description** that combines the most informative elements from the input descriptions, preserving clarity and completeness.
        - Do **not** invent new meanings.
        - Do **not** include redundant or overly narrow wording. 
        """


        normalized_entities_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    normalized_entities_preprompt,
                ),
                (
                    "human",
                    """Here is a list of `Entity` objects considered semantically similar:

                    {entities}

                    Please return a single normalized Entity following the specified format."""
                ),
            ]
        )

        normalized_entities_chain = (
            normalized_entities_prompt | self.llm.with_structured_output(Entity)
        )

        normalized_entities = []
        all_entities_to_normalize = entities.copy()
        similar_entities = {}
        canonical_mapping = {}

        while len(all_entities_to_normalize) > 0:
            ent1 = all_entities_to_normalize.pop(0)
            similar_entities[str(ent1)] = [ent1]
            for ent2 in all_entities_to_normalize:
                similarity_res = purged_entities_chain.invoke(
                    {
                        "term1": ent1.name,
                        "description1": ent1.description,
                        "term2": ent2.name,
                        "description2": ent2.description,
                    }
                )
                if similarity_res.are_similar:
                    similar_entities[str(ent1)].append(ent2)

            if len(similar_entities[str(ent1)]) > 1:
                all_entities_to_normalize = [r for r in all_entities_to_normalize if r not in similar_entities[str(ent1)]]
            
                norm_ent = normalized_entities_chain.invoke(
                        {
                            "entities": similar_entities[str(ent1)],
                        }
                    )
                normalized_entities.append(norm_ent)
                for ent in similar_entities[str(ent1)]:
                    canonical_mapping[str(ent.name)] = norm_ent.name
            else:
                normalized_entities.append(ent1)
                canonical_mapping[str(ent1.name)] = ent1.name
        
        for e in entities:
            if str(e.name) not in canonical_mapping:
                canonical_mapping[str(e.name)] = e.name

        return normalized_entities, canonical_mapping

    def standardize_relations(self, triples: List[Dict], entity_canonical_map):
        """
        Standardize the relations in the triples by ensuring that the head and tail entities are in their canonical forms.
        """

        triples_standard = []
        
        for triple in triples:
            source = triple.head
            target = triple.tail

            try:
                source_canonical = entity_canonical_map[str(source)]
                target_canonical = entity_canonical_map[str(target)]
            except KeyError:
                source_canonical = source
                target_canonical = target
            
            triple.head = source_canonical
            triple.tail = target_canonical

            triples_standard.append(triple)

        return triples_standard
    
    

    def normalize_triples(self, quadruples: List[Triple], entities):
        
        
        purged_quadruples_preprompt = f"""
            You are an expert in semantic comparison of scientific knowledge graph data.

            **TASK**:
            Determine whether two given **quadruples** express the same underlying scientific assertion or fact. Each quadruple consists of:
            - a **head entity**
            - a **relation**
            - a **tail entity**
            - a **description** of the relationship's meaning (i.e., how head + relation + tail should be interpreted)

            **CRITERIA**:
            - Focus on the **semantic meaning** conveyed by the entire quadruple, especially using the **description** to interpret the function of the relation.
            - Consider two quadruples equivalent if they convey the same factual assertion, even when:
                - The direction of the relation is **reversed**, as long as the **semantic meaning** is preserved (e.g., "A uses B" ≡ "B is used by A").
                - The **relation descriptions** indicate they are **logical inverses** with identical intent.
                - The **roles** of head and tail entities are switched, but the overall **assertion remains the same**.

            - Return **"True"** only if the two quadruples are semantically **fully equivalent**, either directly or through logical inversion.
            - Do **not** return "True" if:
                - The relation descriptions suggest differences in intent, specificity, or abstraction.
                - The entities or relations do not align in their conceptual or functional roles, even if similar terms are used.
                - The meaning is only **partially overlapping**, or context-dependent.

            **OUTPUT**:
            - Only output **"True"** if the two quadruples are semantically equivalent.
            - Output **"False"** if they are not.
            - Output strictly **"True"** or **"False"**, with no additional text or explanation. 
            """


        purged_quadruples_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            purged_quadruples_preprompt,
        ),
        (
            "human",
            """Here are two quadruples:

            Quadruple 1:
            - head: {quadruple1_head}
            - relation: {quadruple1_relation}
            - tail: {quadruple1_tail}
            - description: {quadruple1_description}

            Quadruple 2:
            - head: {quadruple2_head}
            - relation: {quadruple2_relation}
            - tail: {quadruple2_tail}
            - description: {quadruple2_description}
            """
                    ),
                ]
            )


        purged_quadruples_chain = (
            purged_quadruples_prompt
            | self.llm.with_structured_output(isSimilar)
        )

        normalized_quadruples_preprompt = """
            You are an expert in normalization of scientific knowledge graph data.

            **TASK**:
            You will receive a list of 'Quadruple' objects semantically similar, each represented by:
            - "head": the head entity
            - "relation": the relation type
            - "tail": the tail entity
            - "description": a short explanation of the relation's meaning
            - "paper_id": the ID of the paper from which the quadruple was extracted

            Your goal is to identify **one representative quadruple** from the list that best captures the meaning of the entire group.

            **INSTRUCTIONS**:
            - Return exactly **one** of the input quadruples — do not create a new one.
            - Select the quadruple whose **head**, **tail**, and **description** are the most **informative**, **precise**, and **representative** of the group's shared semantics.
            - Ensure that the selected quadruple:
                - Uses general, yet scientifically accurate terminology.
                - Preserves the core factual meaning across the group.
                - Avoids redundancy and overly narrow phrasing, while maintaining specificity where relevant.

            Do not include any additional text or explanation in your response. 
            """


        normalized_quadruples_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    normalized_quadruples_preprompt,
                ),
                (
                    "human",
                    """Here is a list of semantically similar quadruples:

                    {quadruples}

                    Please return a single normalized quadruple that best represents the group."""
                ),
            ]
        )

        normalized_quadruples_chain = (
            normalized_quadruples_prompt | self.llm.with_structured_output(Triple)
        )

        normalized_quadruples = []
        all_quadruples_to_normalize = quadruples.copy()
        similar_quadruples = {}

        while len(all_quadruples_to_normalize) > 0:
            quad1 = all_quadruples_to_normalize.pop(0)
            similar_quadruples[str(quad1)] = [quad1]
            for quad2 in all_quadruples_to_normalize:
                similarity_res = purged_quadruples_chain.invoke(
                    {
                        "quadruple1_head": quad1.head,
                        "quadruple1_relation": quad1.relation,
                        "quadruple1_tail": quad1.tail,
                        "quadruple1_description": quad1.description,
                        "quadruple2_head": quad2.head,
                        "quadruple2_relation": quad2.relation, 
                        "quadruple2_tail": quad2.tail,
                        "quadruple2_description": quad2.description,
                    }
                )
                if similarity_res.are_similar:
                    similar_quadruples[str(quad1)].append(quad2)

            if len(similar_quadruples[str(quad1)]) > 1:
                all_quadruples_to_normalize = [r for r in all_quadruples_to_normalize if r not in similar_quadruples[str(quad1)]]
            
                norm_quad = normalized_quadruples_chain.invoke(
                        {
                            "quadruples": similar_quadruples[str(quad1)],
                        }
                    )
                if norm_quad.head in [ent.name for ent in entities] and norm_quad.tail in [ent.name for ent in entities]:
                    normalized_quadruples.append(norm_quad)

            else:
                normalized_quadruples.append(quad1)
        
        return normalized_quadruples
    

    def normalize_triples_list(self, quadruples_list1: List[Triple], quadruples_list2: List[Triple], entities):
    
        purged_quadruples_preprompt = f"""
            You are an expert in semantic comparison of scientific knowledge graph data.

            **TASK**:
            Determine whether two given **quadruples** express the same underlying scientific assertion or fact. Each quadruple consists of:
            - a **head entity**
            - a **relation**
            - a **tail entity**
            - a **description** of the relationship's meaning (i.e., how head + relation + tail should be interpreted)

            **CRITERIA**:
            - Focus on the **semantic meaning** conveyed by the entire quadruple, especially using the **description** to interpret the function of the relation.
            - Consider two quadruples equivalent if they convey the same factual assertion, even when:
                - The direction of the relation is **reversed**, as long as the **semantic meaning** is preserved (e.g., "A uses B" ≡ "B is used by A").
                - The **relation descriptions** indicate they are **logical inverses** with identical intent.
                - The **roles** of head and tail entities are switched, but the overall **assertion remains the same**.

            - Return **"True"** only if the two quadruples are semantically **fully equivalent**, either directly or through logical inversion.
            - Do **not** return "True" if:
                - The relation descriptions suggest differences in intent, specificity, or abstraction.
                - The entities or relations do not align in their conceptual or functional roles, even if similar terms are used.
                - The meaning is only **partially overlapping**, or context-dependent.

            **OUTPUT**:
            - Only output **"True"** if the two quadruples are semantically equivalent.
            - Output **"False"** if they are not.
            - Output strictly **"True"** or **"False"**, with no additional text or explanation. 
            """

        purged_quadruples_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            purged_quadruples_preprompt,
        ),
        (
            "human",
            """Here are two quadruples:

            Quadruple 1:
            - head: {quadruple1_head}
            - relation: {quadruple1_relation}
            - tail: {quadruple1_tail}
            - description: {quadruple1_description}

            Quadruple 2:
            - head: {quadruple2_head}
            - relation: {quadruple2_relation}
            - tail: {quadruple2_tail}
            - description: {quadruple2_description}
            """
                    ),
                ]
            )

        purged_quadruples_chain = (
            purged_quadruples_prompt
            | self.llm.with_structured_output(isSimilar)
        )

        normalized_quadruples_preprompt = """
            You are an expert in normalization of scientific knowledge graph data.

            **TASK**:
            You will receive a list of 'Quadruple' objects semantically similar, each represented by:
            - "head": the head entity
            - "relation": the relation type
            - "tail": the tail entity
            - "description": a short explanation of the relation's meaning
            - "paper_id": the ID of the paper from which the quadruple was extracted

            Your goal is to identify **one representative quadruple** from the list that best captures the meaning of the entire group.

            **INSTRUCTIONS**:
            - Return exactly **one** of the input quadruples — do not create a new one.
            - Select the quadruple whose **head**, **tail**, and **description** are the most **informative**, **precise**, and **representative** of the group's shared semantics.
            - Ensure that the selected quadruple:
                - Uses general, yet scientifically accurate terminology.
                - Preserves the core factual meaning across the group.
                - Avoids redundancy and overly narrow phrasing, while maintaining specificity where relevant.

            Do not include any additional text or explanation in your response. 
            """

        normalized_quadruples_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    normalized_quadruples_preprompt,
                ),
                (
                    "human",
                    """Here is a list of semantically similar quadruples:

                    {quadruples}

                    Please return a single normalized quadruple that best represents the group."""
                ),
            ]
        )

        normalized_quadruples_chain = (
            normalized_quadruples_prompt | self.llm.with_structured_output(Triple)
        )

        similar_quadruples = {}
        normalized_quadruples = []

        for quad1 in quadruples_list1:
            similar_quadruples[str(quad1)] = [quad1]
            quadruples_to_remove = []

            for quad2 in quadruples_list2:
                similarity_res = purged_quadruples_chain.invoke(
                    {
                        "quadruple1_head": quad1.head,
                        "quadruple1_relation": quad1.relation,
                        "quadruple1_tail": quad1.tail,
                        "quadruple1_description": quad1.description,
                        "quadruple2_head": quad2.head,
                        "quadruple2_relation": quad2.relation, 
                        "quadruple2_tail": quad2.tail,
                        "quadruple2_description": quad2.description,
                    }
                )
                if similarity_res.are_similar:

                    similar_quadruples[str(quad1)].append(quad2)
                    quadruples_to_remove.append(quad2)

            if len(quadruples_to_remove) > 0:
                quadruples_list2 = [r for r in quadruples_list2 if r not in quadruples_to_remove]
            
                norm_quad = normalized_quadruples_chain.invoke(
                        {
                            "quadruples": similar_quadruples[str(quad1)],
                        }
                    )
                if norm_quad.head in [ent.name for ent in entities] and norm_quad.tail in [ent.name for ent in entities]:
                    normalized_quadruples.append(norm_quad)
            else:
                normalized_quadruples.append(quad1)

        for quad in quadruples_list2:
            normalized_quadruples.append(quad)

        return normalized_quadruples
    
    def normalize_entities_lists(self, entities_list1: List[Entity], entities_list2: List[Entity]) -> Tuple[List[Entity], Dict[str, str], float]:
        
        purged_entities_preprompt = f"""
        You are an expert in semantic comparison of scientific and technical terminology.

        **TASK**:
        Determine whether two terms refer to the same scientific concept based on their meaning and function.

        **INPUT**:
        You will be given two terms, each with a short description.

        **MATCHING CRITERIA**:
        Return **"True"** only if all the following hold:
        - The two terms describe the **same underlying concept or object**, regardless of surface differences (e.g., abbreviation vs. full form, synonyms, singular/plural).
        - They refer to the **same level of abstraction** and **serve the same function** in scientific discourse.
        - They are **interchangeable** in context without loss of meaning.

        **DO NOT MATCH IF**:
        - One term is a subtype, variation, or implementation of the other.
        - The terms differ in **scope**, **purpose**, or **granularity**.
        - They describe different entities, even if related.

        **POSITIVE EXAMPLES (return "True")**:
        - "BERT" and "BERT model"
        - "Natural Language Processing" and "NLP"
        - "Large language model" and "Large language models"
        - "Convolutional Neural Network" and "CNN"

        **NEGATIVE EXAMPLES (return "False")**:
        - "Transformer" and "BERT" (BERT is a specific kind of Transformer)
        - "Language model" and "Large language model" (different specificity)
        - "Pretrained model" and "BERT" (BERT is an instance of the former)
        - "Word2Vec" and "GloVe" (different methods)

        **OUTPUT**:
        Respond with **only** "True" or "False". No explanations. 
        """

        purged_entities_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", purged_entities_preprompt),
                (
                    "human",
                    """Here are the two terms and their descriptions:

                    Term 1: {term1}
                    Description 1: {description1}

                    Term 2: {term2}
                    Description 2: {description2}
                    """,
                ),
            ]
        )

        purged_entities_chain = purged_entities_prompt | self.llm.with_structured_output(isSimilar)

        normalized_entities_preprompt = """
        You are an expert in entity normalization for scientific documents.

        **TASK**:
        You will receive a list of 'Entity' objects, each defined as:
        - "name": the entity label as found in the text
        - "type": the category of the entity
        - "description": a brief explanation of what the entity refers to

        Your goal is to produce a **single normalized entity** that best represents the entire group.

        **INSTRUCTIONS**:
        - Extract and return **one representative entity** with:
            - A normalized **name** that is clear, specific, and general enough to cover all variations.
            - A **type** that accurately reflects the role or category of the entity across all inputs.
            - A **description** that combines the most informative elements from the input descriptions, preserving clarity and completeness.
        - Do **not** invent new meanings.
        - Do **not** include redundant or overly narrow wording. 
        """

        normalized_entities_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", normalized_entities_preprompt),
                (
                    "human",
                    """Here is a list of `Entity` objects considered semantically similar:

                    {entities}

                    Please return a single normalized Entity following the specified format."""
                ),
            ]
        )

        normalized_entities_chain = normalized_entities_prompt | self.llm.with_structured_output(Entity)

        canonical_mapping = {}
        normalized_entities = []

        entities_list2_copy = entities_list2.copy()

        for ent1 in entities_list1:
            group = [ent1]
            to_remove = []

            for ent2 in entities_list2_copy:
                similarity_res = purged_entities_chain.invoke(
                    {
                        "term1": ent1.name,
                        "description1": ent1.description,
                        "term2": ent2.name,
                        "description2": ent2.description,
                    }
                )
                if similarity_res.are_similar:
                    group.append(ent2)
                    to_remove.append(ent2)

            for r in to_remove:
                entities_list2_copy.remove(r)

            if len(group) > 1:
                norm_ent = normalized_entities_chain.invoke({"entities": group})
                normalized_entities.append(norm_ent)
                for ent in group:
                    canonical_mapping[ent.name] = norm_ent.name
            else:
                normalized_entities.append(ent1)
                canonical_mapping[ent1.name] = ent1.name

        for ent in entities_list2_copy:
            normalized_entities.append(ent)
            canonical_mapping[ent.name] = ent.name

        return normalized_entities, canonical_mapping

        

