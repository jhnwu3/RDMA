from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re
from pathlib import Path

@dataclass
class Entity:
    name: str
    type: str
    start: int
    end: int
    definition: Optional[str] = None

class BiomedicalEntityExtractor:
    def __init__(self, llm_pipeline, ontology_path: Optional[Path] = None):
        """Initialize the entity extractor with LLM pipeline and optional ontology.
        
        Args:
            llm_pipeline: The loaded LLM pipeline for text generation
            ontology_path: Optional path to medical ontology file
        """
        self.pipeline = llm_pipeline
        self.ontology = self._load_ontology(ontology_path) if ontology_path else {}
        
        # Entity types we support
        self.entity_types = {
            "rare_disease": "Diseases affecting < 1/2000 people",
            "disease": "General medical conditions and diseases",
            "symptom_and_sign": "Observable manifestations of diseases",
            "anaphor": "References to previously mentioned rare diseases"
        }
        
    def _load_ontology(self, path: Path) -> Dict:
        """Load medical ontology from jsonl file."""
        ontology = {}
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                for name in entry.get('name', []):
                    ontology[name] = entry.get('definition', '')
        return ontology

    def _construct_extraction_prompt(self, text: str, examples: List[Dict] = None) -> str:
        """Construct prompt for entity extraction."""
        prompt = (
            "Extract medical entities from the following text. For each entity, provide:\n"
            "1. The exact text span\n"
            "2. The entity type (rare_disease, disease, symptom_and_sign, or anaphor)\n"
            "3. The start and end character positions\n\n"
            "Entity type definitions:\n"
        )
        
        # Add entity type definitions
        for etype, definition in self.entity_types.items():
            prompt += f"- {etype}: {definition}\n"
        
        # Add examples if provided
        if examples:
            prompt += "\nExamples:\n"
            for example in examples:
                prompt += f"\nText: {example['text']}\nEntities: {json.dumps(example['entities'], indent=2)}\n"
        
        # Add input text
        prompt += f"\nText to analyze: {text}\n\nExtracted entities (in JSON format):"
        
        return prompt

    def extract_entities(self, text: str, examples: List[Dict] = None) -> List[Entity]:
        """Extract medical entities from text using the LLM.
        
        Args:
            text: Input text to analyze
            examples: Optional list of example extractions for few-shot learning
            
        Returns:
            List of extracted Entity objects
        """
        # Construct prompt
        prompt = self._construct_extraction_prompt(text, examples)
        
        # Get LLM response
        response = self.pipeline(
            prompt,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9
        )
        
        try:
            # Parse JSON response
            extracted = json.loads(response[0]["generated_text"])
            
            # Convert to Entity objects
            entities = []
            for ent in extracted:
                entity = Entity(
                    name=ent['text'],
                    type=ent['type'],
                    start=ent['start'],
                    end=ent['end'],
                    definition=self.ontology.get(ent['text'])
                )
                entities.append(entity)
                
            return entities
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM output: {e}")
            return []

    def validate_entity_positions(self, entity: Entity, text: str) -> bool:
        """Validate that entity positions match the text."""
        if entity.start < 0 or entity.end >= len(text):
            return False
        extracted_text = text[entity.start:entity.end + 1]
        return extracted_text.lower() == entity.name.lower()

    def evaluate_extraction(
        self,
        predicted: List[Entity],
        gold: List[Entity],
        text: str,
        match_type: str = 'exact'
    ) -> Dict[str, float]:
        """Evaluate extraction performance against gold standard.
        
        Args:
            predicted: List of predicted entities
            gold: List of gold standard entities
            text: Original text
            match_type: 'exact' or 'relaxed' matching
            
        Returns:
            Dict containing precision, recall, and F1 scores
        """
        def spans_match(pred: Entity, gold: Entity) -> bool:
            if match_type == 'exact':
                return (pred.start == gold.start and 
                       pred.end == gold.end and 
                       pred.type == gold.type)
            else:  # relaxed matching
                pred_range = range(pred.start, pred.end + 1)
                gold_range = range(gold.start, gold.end + 1)
                return (bool(set(pred_range) & set(gold_range)) and 
                       pred.type == gold.type)

        # Calculate true positives, false positives, and false negatives
        tp = sum(1 for p in predicted 
                for g in gold if spans_match(p, g))
        fp = len(predicted) - tp
        fn = len(gold) - tp

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }

    def analyze_errors(
        self,
        predicted: List[Entity],
        gold: List[Entity],
        text: str
    ) -> Dict[str, List[Dict]]:
        """Analyze extraction errors for error analysis.
        
        Returns dict with false positives and false negatives,
        including surrounding context.
        """
        context_window = 50  # characters of context to include
        
        errors = {
            'false_positives': [],
            'false_negatives': [],
            'type_mismatches': []
        }
        
        # Find false positives and type mismatches
        for pred in predicted:
            found_match = False
            for gold_ent in gold:
                if (pred.start == gold_ent.start and 
                    pred.end == gold_ent.end):
                    if pred.type != gold_ent.type:
                        errors['type_mismatches'].append({
                            'predicted': pred,
                            'gold': gold_ent,
                            'context': text[max(0, pred.start - context_window):
                                           min(len(text), pred.end + context_window)]
                        })
                    found_match = True
                    break
            
            if not found_match:
                errors['false_positives'].append({
                    'entity': pred,
                    'context': text[max(0, pred.start - context_window):
                                   min(len(text), pred.end + context_window)]
                })

        # Find false negatives
        for gold_ent in gold:
            if not any(pred.start == gold_ent.start and 
                      pred.end == gold_ent.end for pred in predicted):
                errors['false_negatives'].append({
                    'entity': gold_ent,
                    'context': text[max(0, gold_ent.start - context_window):
                                   min(len(text), gold_ent.end + context_window)]
                })
        
        return errors