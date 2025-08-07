import requests
import os
import time
import random
from collections import defaultdict
from typing import List, Dict, Optional
import pymupdf
from langchain.prompts import ChatPromptTemplate
from utils.utils import get_llm, load_existing_titles, download_pdf
from utils.classes import PaperMetadata


class KGExpansionAgent:
    """
    Knowledge Graph Expansion Agent for searching and downloading new papers based on existing knowledge graph data.
    """
    
    def __init__(self, model_name: str = "qwen2.5:14b"):
        """
        Initialize the Knowledge Graph Expansion Agent.

        Args:
            model_name: LLM model identifier
        """
        self.llm = get_llm(model_name=model_name)

    def search_new_papers(self, state):
        """
        Search for new papers based on the expansion strategy.
        """

        expansion_strategy = state['expansion_choice']
        
        # Execute different search strategies based on user choice
        if expansion_strategy == "1":
            new_papers = self._search_by_entity_based_search(state['normalized_triples'], state['normalized_entities'], state['download_dir'], state['new_papers_number'])
        elif expansion_strategy == "2":
            metadata = self._get_papers_metadata(state)
            new_papers = self.search_and_download_foundational_papers(metadata, state['download_dir'], state['new_papers_number'])
        elif expansion_strategy == "3":
            metadata = self._get_papers_metadata(state)
            new_papers = self._search_common_citations(metadata, state['download_dir'], state['new_papers_number'])
        else:
            raise ValueError(f"Unsupported strategy: {expansion_strategy}")
        
        return new_papers
    
    
    def search_and_download_papers(self, query, max_pdfs=2, download_dir="new_papers"):
        
        os.makedirs(download_dir, exist_ok=True)
                
        existing_titles = load_existing_titles(download_dir)
        
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        offset = 0
        page_size = 100
        downloaded = 0

        params = {
            "query": query,
            "fields": "title,openAccessPdf,citationCount,isOpenAccess",
            "limit": page_size,
            "offset": offset
        }
        pdf_titles = []
        flag = True
        while flag:
            try:
                response = requests.get(base_url, params=params, timeout=60, verify=True)
                response.raise_for_status()
                flag = False

            except requests.exceptions.RequestException as e:

                backoff_time = random.randint(5,10)
                time.sleep(backoff_time)

        data = response.json().get("data", [])

        papers_ordinati = sorted(data, key=lambda x: x.get("citationCount", 0), reverse=True)

        papers_with_url = [paper for paper in papers_ordinati if paper.get("openAccessPdf", {}).get("url") and paper.get("paperId") not in existing_titles]
        
        for paper in papers_with_url:

            paper_id = paper.get("paperId", "unknown")
            
            pdf_url = paper.get("openAccessPdf", {}).get("url")

            try:
                pdf_path = os.path.join(download_dir, f"{paper_id}.pdf")
                download_pdf(pdf_url, pdf_path)

                downloaded += 1
                pdf_titles.append(pdf_path)
                
            except Exception as e:
                continue  

            time.sleep(random.randint(1, 3)) 

            if downloaded >= max_pdfs:
                break

        return downloaded, pdf_titles
    
    def _get_papers_metadata(self, state):
        """
        Extract metadata from the papers in the state.
        """

        extractor = MetadataExtractor(self.llm)
        papers_metadata = []

        for paper in state['documents']:
            metadata = extractor.extract_metadata_llm(paper)
            
            main_paper = extractor.paper_metadata(metadata, top_n=int(state['new_papers_number'] / len(state['documents'])))
            
            if len(main_paper) > 0:
                
                papers_metadata.append(main_paper)

        return papers_metadata
    
    def search_and_download_foundational_papers(self, papers_metadata, folder_path, new_papers_number):

        foundational_papers = []
        for paper in papers_metadata:
            foundational_papers.extend(paper['foundational_papers'])

        all_downloaded = []
        for title in foundational_papers:
            _, pdf_titles = self.search_and_download_papers(query=title, max_pdfs=1, download_dir=folder_path)
            all_downloaded.extend(pdf_titles)
            if len(all_downloaded) >= new_papers_number:
                break

        return all_downloaded
        
    def _search_by_entity_based_search(self, triples, entities, download_dir, new_papers_number):
        """
        Search for new papers using entity-based search strategy.
    
        """ 
        entity_degree = {entity.name: 0 for entity in entities}

        for triple in triples:
            head = triple.head
            tail = triple.tail
            entity_degree[head] += 1
            entity_degree[tail] += 1
            
        sorted_entities = sorted(entity_degree.items(), key=lambda x: x[1], reverse=True)[:5]

        entities_top = [k for k, _ in sorted_entities]

        base_query = " ".join(entities_top)
        queries = [base_query]
        
        for i in range(len(entities_top)):
            remaining_entities = entities_top[:i] + entities_top[i+1:]
            query = " ".join(remaining_entities)
            queries.append(query)
    
        downloaded_total = 0
        max_pdfs = new_papers_number
        all_downloaded = []

        for query in queries:
            if downloaded_total >= max_pdfs:
                break

            remaining = max_pdfs - downloaded_total
            print(f"Searching for papers with query: '{query}'")
            downloaded, pdf_titles = self.search_and_download_papers(
                query=query,
                max_pdfs=remaining,
                download_dir=download_dir
            )

            downloaded_total += downloaded
            all_downloaded.extend(pdf_titles)

        return all_downloaded

    
    def _search_common_citations(self, papers_metadata, folder_path, new_papers_number):
        """
        Search for common citations across multiple papers and download them.
        """

        doi_index = defaultdict(list)
        
        for paper in papers_metadata:
            for ref in paper['references_metadata']:
                paper_name = ref.get('title', '')
                doi = ref.get('doi', '')

                if doi:
                    doi_index[doi].append({
                        'name': paper_name,
                    })

        common_references = set()
        
        for doi, refs in doi_index.items():
            if len(refs) > 1:
                common_references.add(refs[0]['name'])
        
        common_references = list(common_references)

        downloaded = []
        for title in common_references:
            _, pdf_titles = self.search_and_download_papers(query=title, max_pdfs=1, download_dir=folder_path)
            downloaded.extend(pdf_titles)
            if len(downloaded) >= new_papers_number:
                break

        return downloaded


class MetadataExtractor:
    """
    Interacts with the OpenAlex API to retrieve paper metadata and foundational papers.
    """
    
    def __init__(self, llm):
        self.openalex_base = "https://api.openalex.org"
        self.llm = llm
    
    def clean_doi(self, doi: str) -> str:
        if 'doi.org/' in doi:
            doi = doi.split('doi.org/')[-1]
        return doi.strip()
    
    def get_paper_from_doi(self, doi: str) -> Optional[Dict]:
        """
        Retrieves paper information from OpenAlex using the DOI.
        """
        doi = self.clean_doi(doi)
        url = f"{self.openalex_base}/works/doi:{doi}"
        
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                citation_normalized_percentile = data.get('citation_normalized_percentile', {})
                if citation_normalized_percentile:
                    citation_normalized_percentile = citation_normalized_percentile.get('value', 0)
                else:
                    citation_normalized_percentile = 0

                return {
                    'id': data['id'],
                    'doi': doi,
                    'title': data.get('title', 'N/A'),
                    'authors': [author['author']['display_name'] 
                               for author in data.get('authorships', [])],
                    'year': data.get('publication_year'),
                    'references': data.get('referenced_works', []),
                    'cited_by_count': data.get('cited_by_count', 0),
                    'citation_normalized_percentile': citation_normalized_percentile,
                }
        except Exception as e:
            print(f"Error retrieving paper with DOI {doi}: {e}")
            return None
    
    def get_paper_by_id(self, work_id: str) -> Optional[Dict]:
        """
        Retrieves paper information from OpenAlex using the work ID.
        """
        if work_id.startswith('https://openalex.org/'):
            work_id = work_id.split('/')[-1]
        
        url = f"{self.openalex_base}/works/{work_id}"
        
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                citation_normalized_percentile = data.get('citation_normalized_percentile', {})
                if citation_normalized_percentile:
                    citation_normalized_percentile = citation_normalized_percentile.get('value', 0)
                else:
                    citation_normalized_percentile = 0

                return {
                    'id': data['id'],
                    'doi': data.get('doi', ''),
                    'title': data.get('title', 'N/A'),
                    'authors': [author['author']['display_name'] 
                               for author in data.get('authorships', [])],
                    'year': data.get('publication_year'),
                    'references': data.get('referenced_works', []),
                    'cited_by_count': data.get('cited_by_count', 0),
                    'citation_normalized_percentile': citation_normalized_percentile,
                }
        except Exception as e:
            print(f"Error retrieving paper with ID {work_id}: {e}")
        
        return None
    
    def get_paper_by_title(self, title: str) -> Optional[Dict]:

        """
        Retrieves paper information from OpenAlex using the title.
        """
        
        url = "https://api.openalex.org/works"

        params = {
            "filter": f"title.search:{title}",
            "per-page": 1  
        }

        try:
            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()['results'][0]
                citation_normalized_percentile = data.get('citation_normalized_percentile', {})
                if citation_normalized_percentile:
                    citation_normalized_percentile = citation_normalized_percentile.get('value', 0)
                else:
                    citation_normalized_percentile = 0

                return {
                        'id': data['id'],
                        'doi': data.get('doi', ''),
                        'title': data.get('title', 'N/A'),
                        'authors': [author['author']['display_name'] 
                                for author in data.get('authorships', [])],
                        'year': data.get('publication_year'),
                        'references': data.get('referenced_works', []),
                        'cited_by_count': data.get('cited_by_count', 0),
                        'citation_normalized_percentile': citation_normalized_percentile,
                    }
            
        except Exception as e:
            print(f"Error retrieving paper with title '{title}': {e}")
            return None
    

    def extract_metadata_llm(self, pdf_path: str) -> tuple[Optional[str], Optional[list]]:
        """
        Use an LLM to extract metadata from a scientific PDF text.
        """

        doc = pymupdf.open(pdf_path)

        text = doc[0].get_text()

        
        system_prompt = """
            You are an expert in extracting metadata from scientific papers. You will receive a text extracted from a PDF document.
            Your task is to extract the title, authors of the paper and the DOI, if present. If you find a DOI, return it in the format '10.xxxx/xxxxxx'. 
            Answer in JSON format with the following structure:
            
                "title": "The title of the paper",
                "authors": "List of authors separated by commas"
                "doi": "10.xxxx/xxxxxx" (if available)
        
            """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Text extract from PDF:\n\n {text}")
            ]
        )

        chain = prompt | self.llm.with_structured_output(PaperMetadata)
        response = chain.invoke({"text": text})

        return {"title": response.title, "authors": response.authors, "doi": response.doi}

    
    def paper_metadata(self, metadata: str, top_n: int = 1) -> List[Dict]:
        """
        Gets the metadata of a paper and its foundational papers based on the provided information.
        """
        
        main_paper = self.get_paper_from_doi(metadata['doi'])
        if not main_paper:
            main_paper = self.get_paper_by_title(metadata['title'])
        
        direct_references = {} 
        main_paper['references_metadata'] = []
        main_paper['foundational_papers'] = []

        for ref_id in main_paper['references']:
            ref_paper = self.get_paper_by_id(ref_id)

            if ref_paper :
                direct_references[ref_id] = ref_paper
                main_paper['references_metadata'].append(ref_paper)

        ref_percentile = {}

        for ref_id, paper in direct_references.items():
            
            ref_percentile[ref_id] = paper.get('citation_normalized_percentile')

        sorted_references = sorted(ref_percentile.items(), key=lambda x: x[1], reverse=True)
        top_references = sorted_references[:top_n]
        

        for ref_id, _ in top_references:
            paper = direct_references.get(ref_id)
            main_paper['foundational_papers'].append(paper)

        return main_paper
