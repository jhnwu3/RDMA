import json
import spacy
from typing import Dict, List, Set, Tuple
from outdated_scripts.ontology import OntologyConverter
from scispacy.abbreviation import AbbreviationDetector
from negspacy.negation import Negex
from scispacy.linking import EntityLinker
spacy.require_gpu(gpu_id=0)
class ClinicalTextExtractor:
    def __init__(self, rare_disease_triples_path: str):
        """
        Initialize the extractor with path to RareDisease_Phenotype_Triples.json
        
        Args:
            rare_disease_triples_path (str): Path to the triples JSON file
        """
        # Initialize mapping dictionaries
        self.phenotype_map = {}  # name -> HPO ID
        self.disease_map = {}    # name -> ORPHA ID
        self._load_mappings(rare_disease_triples_path)

    def _load_mappings(self, filepath: str) -> None:
        """Load phenotype and disease mappings from triples file"""
        with open(filepath) as f:
            triples = json.load(f)
            for triple in triples:
                # Store all terms in lowercase for consistent matching
                for name in triple['target']['name']:
                    self.phenotype_map[name.lower().strip()] = triple['target']['id']
                for name in triple['source']['name']:
                    self.disease_map[name.lower().strip()] = triple['source']['id']

    def _check_term_boundaries(self, term: str, note: str) -> bool:
        """
        Check if term exists as a complete word in the note
        
        Args:
            term (str): Term to look for
            note (str): Full clinical note
            
        Returns:
            bool: True if term exists with proper word boundaries
        """
        boundaries = [
            f" {term} ", f" {term},", f" {term}.", 
            f" {term}!", f"*{term})", f"({term}*",
            f" {term}\n", f" {term}:"  # Additional boundaries
        ]
        return any(bound in f" {note} " for bound in boundaries)

    def process_text(self, text: str) -> Dict[str, List[Dict]]:
        """
        Process a clinical note and extract phenotypes and diseases using pattern matching
        
        Args:
            clinical_note (str): The clinical note text
            
        Returns:
            Dict with two keys:
            - 'phenotypes': List[Dict] with hpoID and hpoName
            - 'diseases': List[Dict] with OrphaID and ProblemName
        """
        # Preprocess note
        processed_text = text.lower().replace("dr.", "doctor")
        
        phenotypes = set()  # Using set to avoid duplicates
        diseases = set()
        
        # Check for phenotypes
        for pt_name in self.phenotype_map:
            if pt_name in processed_text and self._check_term_boundaries(pt_name, processed_text):
                # Special handling for ambiguous terms
                if pt_name == "ra" and any(m in text for m in ["in ra", "on ra"]):
                    continue
                phenotypes.add((self.phenotype_map[pt_name], pt_name))
        
        # Check for diseases
        undesirable_terms = {"has", "he", "hi", "md", "med", "pale", "ped", "peds", "plan", "pm"}
        for disease_name in self.disease_map:
            if disease_name in processed_text and self._check_term_boundaries(disease_name, processed_text):
                # Skip if contains undesirable terms
                if any(term in disease_name for term in undesirable_terms):
                    continue
                diseases.add((self.disease_map[disease_name], disease_name))
        
        return {
            'phenotypes': [
                {'hpoID': hpo_id, 'hpoName': name}
                for hpo_id, name in phenotypes
            ],
            'diseases': [
                {'OrphaID': orpha_id, 'ProblemName': name}
                for orpha_id, name in diseases
            ]
        }

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Any
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from negspacy.termsets import termset
import json

@dataclass
class ExtractedConcept:
    """Store information about an extracted concept."""
    code: str  # HPO or ORPHA code
    name: str  # Original text or entity name
    source: str  # 'direct_match', 'umls_conversion', or 'text_matching'
    confidence: float  # Confidence score (1.0 for direct matches)
    ontology_type: str  # 'HPO' or 'ORPHA'
    umls_cui: Optional[str] = None  # UMLS CUI if derived through UMLS

class CombinedOntologyPipeline:
    def __init__(
        self,
        ontology_converter,
        rare_disease_triples_path: str,
        spacy_model: str = "en_core_sci_scibert",
        gpu_id: Optional[int] = None
    ):
        """
        Initialize the combined pipeline.
        
        Args:
            ontology_converter: Instance of OntologyConverter
            rare_disease_triples_path: Path to the rare disease triples JSON file
            spacy_model: Name of the spaCy model to use
            gpu_id: GPU ID to use (None for CPU)
        """
        # Initialize components
        self.ontology_converter = ontology_converter
        self.text_extractor = ClinicalTextExtractor(rare_disease_triples_path)
        
        # Initialize spaCy pipeline
        if gpu_id is not None:
            spacy.require_gpu(gpu_id=gpu_id)
        self.nlp = self._setup_spacy_pipeline(spacy_model)
    
    def _setup_spacy_pipeline(self, model_name: str) -> Language:
        """Set up the spaCy pipeline with necessary components."""
        nlp = spacy.load(model_name)
        ts = termset("en_clinical_sensitive")
        ts.add_patterns(
            {
                "preceding_negations": ["deny", "refuse", "neither", "nor", "call for", "call 911", "return if", "seek immediate medical attention", "seek medical", "notify care team"],
                "following_negations": ["absent", "deny", "decline", "seek medical attention", "seek medical", "notify care team"]
            }
        )

        # Add pipeline components
        nlp.add_pipe("abbreviation_detector")
        nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": True,
                "linker_name": "umls",
                "filter_for_definitions": False
            }
        )
        nlp.add_pipe(
            "negex",
            config={
                "neg_termset": ts.get_patterns()
            }
        )
                
        # Set maximum text length
        nlp.max_length = 3000000
        
        return nlp
    
    def _extract_umls_entities(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract entities and their UMLS CUIs from spaCy doc."""
        entities = []
        for ent in doc.ents:
            if ent._.kb_ents:  # Only process entities with KB links
                # Get the best matching CUI
                cui, score = max(ent._.kb_ents, key=lambda x: x[1])
                entities.append({
                    'text': ent.text,
                    'cui': cui,
                    'score': score
                })
        return entities
    
    def _convert_umls_to_ontologies(
        self, 
        umls_entities: List[Dict[str, Any]]
    ) -> List[ExtractedConcept]:
        """Convert UMLS entities to HPO/ORPHA concepts."""
        concepts = []
        
        for entity in umls_entities:
            # Get mappings for this UMLS CUI
            mappings = self.ontology_converter.check_and_convert_umls(entity['cui'])
            
            # Add HPO mappings
            for hpo_meta in mappings['hpo_metadata']:
                concepts.append(ExtractedConcept(
                    code=hpo_meta['code'],
                    name=entity['text'],
                    source='umls_conversion',
                    confidence=entity['score'],
                    ontology_type='HPO',
                    umls_cui=entity['cui']
                ))
            
            # Add ORPHA mappings
            for orpha_meta in mappings['orpha_metadata']:
                concepts.append(ExtractedConcept(
                    code=orpha_meta['code'],
                    name=entity['text'],
                    source='umls_conversion',
                    confidence=entity['score'],
                    ontology_type='ORPHA',
                    umls_cui=entity['cui']
                ))
        
        return concepts
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess clinical text for matching."""
        return text.lower().replace("dr.", "doctor")

    def _check_term_boundaries(self, term: str, note: str) -> bool:
        """
        Check if term exists as a complete word in the note with proper boundaries.
        
        Args:
            term (str): Term to look for
            note (str): Full clinical note
            
        Returns:
            bool: True if term exists with proper word boundaries
        """
        boundaries = [
            f" {term} ", f" {term},", f" {term}.", 
            f" {term}!", f"*{term})", f"({term}*",
            f" {term}\n", f" {term}:"
        ]
        return any(bound in f" {note} " for bound in boundaries)

    def _direct_text_matching(self, text: str) -> List[ExtractedConcept]:
        """Extract concepts using direct text matching with careful preprocessing."""
        processed_text = self._preprocess_text(text)
        concepts = []
        
        # Check for phenotypes
        for pt_name, hpo_id in self.text_extractor.phenotype_map.items():
            if pt_name in processed_text and self._check_term_boundaries(pt_name, processed_text):
                # Special handling for ambiguous terms
                if pt_name == "ra" and any(m in processed_text for m in ["in ra", "on ra"]):
                    continue

                concepts.append(ExtractedConcept(
                    code=hpo_id,
                    name=pt_name,
                    source='direct_match',
                    confidence=1.0,
                    ontology_type='HPO'
                ))
        
        # Check for diseases
        undesirable_terms = {"has", "he", "hi", "md", "med", "pale", "ped", "peds", "plan", "pm"}
        for disease_name, orpha_id in self.text_extractor.disease_map.items():
            if disease_name in processed_text and self._check_term_boundaries(disease_name, processed_text):
                # Skip if contains undesirable terms
                if any(term in disease_name.split() for term in undesirable_terms):
                    continue
                    
                concepts.append(ExtractedConcept(
                    code=orpha_id,
                    name=disease_name,
                    source='direct_match',
                    confidence=1.0,
                    ontology_type='ORPHA'
                ))
        
        return concepts
    
    def _additional_text_matching(
        self, 
        text: str, 
        entities: List[Dict[str, Any]]
    ) -> List[ExtractedConcept]:
        """Perform additional text matching on extracted entities."""
        concepts = []
        
        # Create a set of already processed entity texts
        processed_texts = {ent['text'].lower() for ent in entities}
        
        # Preprocess text once
        processed_full_text = self._preprocess_text(text)
        
        # For each entity, try text matching on its components
        for entity in entities:
            entity_text = entity['text']
            
            # Split multi-word entities and check each part
            words = entity_text.split()
            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    substring = " ".join(words[i:j]).lower()
                    
                    # Skip if already processed or if it contains undesirable terms
                    if substring in processed_texts:
                        continue
                        
                    # Skip substrings containing undesirable terms
                    if any(term in substring.split() for term in {"has", "he", "hi", "md", "med", 
                                                                "pale", "ped", "peds", "plan", "pm"}):
                        continue
                        
                    # Only process if the substring appears with proper boundaries
                    if not self._check_term_boundaries(substring, processed_full_text):
                        continue
                    
                    # Try text matching on this substring
                    results = self.text_extractor.process_text(substring)
                    
                    # Add any new matches
                    for phenotype in results['phenotypes']:
                        concepts.append(ExtractedConcept(
                            code=phenotype['hpoID'],
                            name=phenotype['hpoName'],
                            source='text_matching',
                            confidence=0.8,  # Lower confidence for substring matches
                            ontology_type='HPO'
                        ))
                    
                    for disease in results['diseases']:
                        concepts.append(ExtractedConcept(
                            code=disease['OrphaID'],
                            name=disease['ProblemName'],
                            source='text_matching',
                            confidence=0.8,
                            ontology_type='ORPHA'
                        ))
        
        return concepts
    
    def process_text(self, text: str) -> List[ExtractedConcept]:
        """
        Process clinical text through the combined pipeline.
        
        Args:
            text: The clinical text to process
            
        Returns:
            List of ExtractedConcept objects with all findings
        """
        # Step 1: Direct text matching
        concepts = self._direct_text_matching(text)
        
        # Step 2: NER -> UMLS -> HPO/ORPHA
        doc = self.nlp(text)
        umls_entities = self._extract_umls_entities(doc)
        concepts.extend(self._convert_umls_to_ontologies(umls_entities))
        
        # Step 3: Additional text matching on entities
        concepts.extend(self._additional_text_matching(text, umls_entities))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in concepts:
            key = (concept.code, concept.ontology_type)
            if key not in seen:
                seen.add(key)
                unique_concepts.append(concept)
        
        return unique_concepts
    
    def format_results(self, concepts: List[ExtractedConcept]) -> Dict[str, List[Dict[str, Any]]]:
        """Format the results into a structured dictionary."""
        return {
            'hpo_terms': [
                {
                    'id': c.code,
                    'name': c.name,
                    'source': c.source,
                    'confidence': c.confidence,
                    'umls_cui': c.umls_cui
                }
                for c in concepts if c.ontology_type == 'HPO'
            ],
            'orpha_terms': [
                {
                    'id': c.code,
                    'name': c.name,
                    'source': c.source,
                    'confidence': c.confidence,
                    'umls_cui': c.umls_cui
                }
                for c in concepts if c.ontology_type == 'ORPHA'
            ]
        }

# Example usage
if __name__ == "__main__":
    # Initialize components
    converter = OntologyConverter(
        "export/ontology/hpo/hp.json",
        "export/ontology/ordo/ORDO2_UMLS_ICD.json"
    )
    
    # Initialize combined pipeline
    pipeline = CombinedOntologyPipeline(
        ontology_converter=converter,
        rare_disease_triples_path="RareDisease_Phenotype_Triples.json",
        gpu_id=0
    )
    
    # Example clinical note
    text = """
    Patient presents with severe joint pain and rheumatoid arthritis.
    History suggests possible familial Mediterranean fever.
    No signs of lupus or other autoimmune conditions.
    """
    
    # Process text
    concepts = pipeline.process_text(text)
    
    # Format and print results
    results = pipeline.format_results(concepts)
    print(json.dumps(results, indent=2))