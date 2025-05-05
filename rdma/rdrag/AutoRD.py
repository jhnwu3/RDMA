import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from difflib import get_close_matches
import re

@dataclass
class Entity:
    name: str  # Original extracted term
    start: int
    end: int
    entity_type: str
    negated: bool = False
    orpha_id: Optional[str] = None
    confidence: float = 1.0
    extracted_phrase: str = ""  # The exact phrase from text
    matched_orpha_name: Optional[str] = None  # The name from Orphanet if matched

class SimpleAutoRD:
    def __init__(
        self,
        pipeline: Any,
        rare_disease_ontology_path: str,
    ):
        self.pipeline = pipeline
        self.rare_disease_ontology = self._load_jsonl(rare_disease_ontology_path)
        # Create a mapping of disease names to their ontology entries
        self.disease_mapping = {}
        for entry in self.rare_disease_ontology:
            # Keep the original ID format but also add a normalized version
            normalized_entry = {
                'name': entry['name'],
                'id': entry['id'],  # Keep original format
                'orpha_id': entry['id'],  # Add normalized version
                'definition': entry.get('definition', ''),
            }
            self.disease_mapping[entry['name'].lower()] = normalized_entry


    
    
    def inspect_ontology(self) -> None:
        """Print the structure and statistics of the loaded ontology file."""
        print("\nOntology Structure Analysis:")
        print("-" * 50)
        
        # Basic statistics
        print(f"Total entries: {len(self.rare_disease_ontology)}")
        
        # Analyze structure of first few entries
        if self.rare_disease_ontology:
            print("\nSample entry structure:")
            sample_entry = self.rare_disease_ontology[0]
            for key, value in sample_entry.items():
                print(f"  {key}: {type(value).__name__} = {value[:100] if isinstance(value, str) else value}")
        
        # Analyze key consistency
        all_keys = set()
        key_counts = {}
        
        for entry in self.rare_disease_ontology:
            entry_keys = set(entry.keys())
            all_keys.update(entry_keys)
            
            # Count how many entries have each key
            for key in entry_keys:
                key_counts[key] = key_counts.get(key, 0) + 1
        
        print("\nKey statistics:")
        total_entries = len(self.rare_disease_ontology)
        for key in sorted(all_keys):
            count = key_counts[key]
            percentage = (count / total_entries) * 100
            print(f"  {key}: present in {count}/{total_entries} entries ({percentage:.1f}%)")
        
        # Check for potential issues
        print("\nPotential issues:")
        required_keys = {'name', 'id'}  # Updated required keys
        missing_required = [entry for entry in self.rare_disease_ontology 
                        if not all(key in entry for key in required_keys)]
        
        if missing_required:
            print(f"- {len(missing_required)} entries missing required keys")
            print("  Sample problematic entry:")
            print(f"  {missing_required[0]}")
        
        # Value analysis for critical fields
        print("\nValue analysis:")
        print("Name statistics:")
        empty_names = sum(1 for entry in self.rare_disease_ontology if not entry.get('name'))
        print(f"- Empty names: {empty_names}")
        
        print("\nORPHA ID statistics:")
        orpha_ids = [entry['id'] for entry in self.rare_disease_ontology if 'id' in entry]
        malformed_ids = [id for id in orpha_ids if not id.startswith('ORPHA:')]
        print(f"- Total ORPHA IDs: {len(orpha_ids)}")
        print(f"- Malformed ORPHA IDs: {len(malformed_ids)}")
        if malformed_ids:
            print(f"  Sample malformed ID: {malformed_ids[0]}")
            
        print("\nDefinition statistics:")
        has_definition = sum(1 for entry in self.rare_disease_ontology if entry.get('definition'))
        print(f"- Entries with definitions: {has_definition}/{total_entries} ({(has_definition/total_entries)*100:.1f}%)")
        
    def _load_jsonl(self, path: str) -> List[Dict]:
        with open(path) as f:
            return [json.loads(line) for line in f]

    def _generate_text(self, prompt: str) -> str:
        """Generate text using the pipeline with chat template."""
        messages = [
            {
                "role": "system", 
                "content": "You are an expert in healthcare and biomedical domain. Extract medical entities accurately."
            },
            {
                "role": "user", 
                "content": prompt
            },
        ]
        
        full_prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self.pipeline(
            full_prompt,
            max_new_tokens=20000, # 20k new tokens to see full response in case it's that long.
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
        
        return outputs[0]["generated_text"][len(full_prompt):]

    def _extract_terms(self, text: str) -> List[str]:
        """Extract medical terms from text, returning a simple list of terms."""
        prompt = f"""Extract all diseases and conditions that are NOT negated (i.e., don't include terms that are preceded by 'no', 'not', 'without', etc.) from the text below.

            Text: {text}

            Return only a Python list of strings, with each term exactly as it appears in the text."""
        
        try:
            response = self._generate_text(prompt).strip()
            # print("LLM DEBUG:", response)
            
            # Extract content between square brackets if present
            if '[' in response and ']' in response:
                response = response[response.find('[') + 1:response.rfind(']')]
            
            # Split on commas and clean up each term
            terms = []
            for term in response.split(','):
                # Clean up the term
                cleaned_term = term.strip()
                # Remove surrounding quotes (single or double)
                cleaned_term = cleaned_term.strip('"').strip("'")
                # Only add non-empty terms
                if cleaned_term:
                    terms.append(cleaned_term)
            
            # print("TERMS DEBUG:", terms)
            return terms
                
        except Exception as e:
            print(f"Error in term extraction: {str(e)}")
            return []

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity score using a combination of metrics."""
        from difflib import SequenceMatcher
        
        # Convert to lowercase for comparison
        s1, s2 = s1.lower(), s2.lower()
        
        # Get word sets
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        # Calculate Jaccard similarity for words
        word_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
        
        # Calculate sequence similarity
        seq_similarity = SequenceMatcher(None, s1, s2).ratio()
        
        # Combine both metrics (weighing sequence similarity more heavily)
        return (0.3 * word_similarity + 0.7 * seq_similarity)

    def _verify_match_with_llm(self, term: str, candidates: List[Tuple[str, float]]) -> Optional[str]:
        """Use LLM to verify the best match among candidates."""
        if not candidates:
            return None
            
        candidates_str = "\n".join(f"- {name} (similarity: {score:.2f})" 
                                 for name, score in candidates)
        
        prompt = f"""Given a medical term and potential matches from our ontology, determine if any are valid matches.

Term: {term}

Potential matches:
{candidates_str}

Respond with only the name of the best matching term, or "none" if none are valid matches."""

        try:
            response = self._generate_text(prompt).strip().lower()
            if response == "none":
                return None
            # Find the candidate that most closely matches the LLM's response
            matches = get_close_matches(response, [c[0] for c in candidates], n=1, cutoff=0.9)
            return matches[0] if matches else None
        except Exception:
            return candidates[0][0] if candidates else None
        
    def _split_text(self, text: str, max_chunk_size: int = 4000) -> List[str]:
        """Split text into chunks, ensuring splits occur at delimiters.
        
        Args:
            text: Input text to split
            max_chunk_size: Maximum size of each chunk before forcing a split
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = ""
        
        # Define preferred split points in order of preference
        delimiters = ["\n\n", "\n", ". ", "; ", ", "]
        
        # Helper function to find the best delimiter position
        def find_split_point(text: str, max_size: int) -> int:
            if len(text) <= max_size:
                return len(text)
                
            # Try each delimiter in order of preference
            for delimiter in delimiters:
                # Look for the last delimiter before max_size
                pos = text[:max_size].rfind(delimiter)
                if pos > 0:  # Found a good split point
                    return pos + len(delimiter)
                    
            # If no good delimiter found, force split at max_size
            return max_size
        
        while text:
            split_point = find_split_point(text, max_chunk_size)
            
            if split_point == len(text):
                chunks.append(text)
                break
                
            # Add chunk and continue with remaining text
            chunks.append(text[:split_point].strip())
            text = text[split_point:].strip()
        
        return [c for c in chunks if c]  # Remove any empty chunks
        
    def process_text(self, text: str) -> List[Entity]:
        """Process text through the pipeline."""
        # Split text into natural chunks
        chunks = self._split_text(text)
        
        # Process each chunk and combine results
        all_entities = []
        for chunk in chunks:
            try:
                chunk_entities = self._process_chunk(chunk)
                all_entities.extend(chunk_entities)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
                
        return all_entities
    
    def _fuzzy_match_disease(self, term: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Find closest matching diseases in ontology using fuzzy matching.
        
        Returns:
            List of tuples (disease_name, similarity_score) above threshold
        """
        term = term.lower()
        matches = []
        
        # First try exact word set matching
        term_words = set(term.split())
        
        for disease_name in self.disease_mapping.keys():
            # Calculate similarity score
            similarity = self._string_similarity(term, disease_name)
            if similarity >= threshold:
                matches.append((disease_name, similarity))
        
        # Sort by similarity score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:5]  # Return top 5 matches


    def _verify_rare_disease_with_llm(self, term: str, ontology_entry: Optional[Dict] = None) -> Tuple[bool, float]:
        """Verify if the term is actually a rare disease using LLM.
        
        Args:
            term: The extracted term to verify
            ontology_entry: Optional ontology entry if available
            
        Returns:
            Tuple[bool, float]: (is_rare_disease, confidence)
        """
        context = ""
        if ontology_entry:
            context = f"\nOrphanet information:\nDisease: {ontology_entry['name']}\nOrpha ID: {ontology_entry['orpha_id']}"
        
        prompt = f"""Determine if the following medical term represents a rare disease.

        Term: {term}{context}

        Consider:
        1. Is this a disease (not just a symptom or condition)?
        2. Is it rare (affecting less than 1 in 2000 people)?
        3. If an Orphanet entry is provided, does the term actually match it?
        4. If the orphanet diseases have rare in its name, make sure that the extracted term contains the word rare for it to be considered a rare disease.
        5. Specific forms/variants of common diseases are not rare diseases unless explicitly stated as rare in the term and context.
        
        Respond with only one line in this format:
        DECISION: true/false

        Example responses:
        DECISION: true
        DECISION: false"""

        try:
            response = self._generate_text(prompt).strip().lower()
            # Extract decision
            match = re.search(r'decision:\s*(true|false)', response)
            if match:
                return match.group(1) == 'true'
            return False
        except Exception as e:
            print(f"Error in LLM verification: {str(e)}")
            return False

    def _process_chunk(self, text: str) -> List[Entity]:
        """Process a single chunk of text."""
        try:
            extracted_terms = self._extract_terms(text)
            
            # Deduplicate terms while preserving order
            seen = set()
            unique_terms = []
            for term in extracted_terms:
                if term.lower() not in seen:
                    seen.add(term.lower())
                    unique_terms.append(term)
            
            entities = []
            for term in unique_terms:
                try:
                    # Find the exact phrase in text
                    term_pattern = re.escape(term)
                    matches = list(re.finditer(term_pattern, text, re.IGNORECASE))
                    if not matches:
                        continue
                        
                    # First try exact matching
                    ontology_entry = self.disease_mapping.get(term.lower())
                    matched_name = None
                    
                    if not ontology_entry:
                        # If no exact match, try fuzzy matching
                        candidates = self._fuzzy_match_disease(term)
                        if candidates:
                            # Only verify with LLM if we have close matches
                            if candidates[0][1] > 0.9:  # High confidence match
                                best_match = candidates[0][0]
                                ontology_entry = self.disease_mapping[best_match]
                                matched_name = best_match
                            else:
                                # Verify with LLM for less confident matches
                                best_match = self._verify_match_with_llm(term, candidates)
                                if best_match:
                                    ontology_entry = self.disease_mapping[best_match]
                                    matched_name = best_match
                    
                    # Verify if it's actually a rare disease using LLM
                    is_rare = self._verify_rare_disease_with_llm(term, ontology_entry)
                    
                    # Only create entity if LLM confirms it's a rare disease
                    if is_rare:
                        for match in matches:
                            entity = Entity(
                                name=term,
                                start=match.start(),
                                end=match.end(),
                                entity_type='rare_disease',
                                orpha_id=ontology_entry.get('orpha_id') if ontology_entry else None,
                                extracted_phrase=match.group(),
                                matched_orpha_name=matched_name or (ontology_entry['name'] if ontology_entry else None)
                            )
                            entities.append(entity)
                            
                except Exception as e:
                    print(f"Error processing term '{term}': {str(e)}")
                    continue
                    
            return entities
        except Exception as e:
            print(f"Error in _process_chunk: {str(e)}")
            return []
        
        