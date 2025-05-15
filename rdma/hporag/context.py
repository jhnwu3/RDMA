import re
from typing import List, Dict, Optional, Union, Any
from datetime import datetime


class ContextExtractor:
    """
    Extracts relevant context for entities from clinical notes.

    This class is responsible for finding the most relevant sentences or sections
    in a clinical note that provide context for extracted entities. It can be used
    by different components in the pipeline to ensure consistent context extraction.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the context extractor.

        Args:
            debug: Whether to print debug information
        """
        self.debug = debug

    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")

    def extract_sentences(self, text: str) -> List[str]:
        """
        Split clinical text into sentences.

        Args:
            text: Clinical note text

        Returns:
            List of sentences extracted from the text
        """
        # First split by common sentence terminators while preserving them
        sentence_parts = []
        for part in re.split(r"([.!?])", text):
            if part.strip():
                if part in ".!?":
                    if sentence_parts:
                        sentence_parts[-1] += part
                else:
                    sentence_parts.append(part.strip())

        # Then handle other clinical note delimiters like line breaks and semicolons
        sentences = []
        for part in sentence_parts:
            # Split by semicolons and newlines
            for subpart in re.split(r"[;\n]", part):
                if subpart.strip():
                    sentences.append(subpart.strip())

        self._debug_print(f"Extracted {len(sentences)} sentences from text")
        return sentences

    def find_entity_context(
        self, entity: str, sentences: List[str], window_size: int = 0
    ) -> Optional[str]:
        """
        Find the most relevant context for a given entity.

        Args:
            entity: Entity to find context for
            sentences: List of sentences from the clinical note
            window_size: Number of additional sentences to include before and after the matching sentence

        Returns:
            The most relevant context or None if no match found
        """
        entity_lower = entity.lower()

        # Try exact matching first
        for i, sentence in enumerate(sentences):
            if entity_lower in sentence.lower():
                # Found exact match - include surrounding sentences based on window_size
                return self._get_context_window(sentences, i, window_size)

        # If no exact match, try fuzzy matching based on word overlap
        entity_words = set(re.findall(r"\b\w+\b", entity_lower))
        if not entity_words:
            return None

        best_match_index = -1
        best_score = 0

        for i, sentence in enumerate(sentences):
            sentence_words = set(re.findall(r"\b\w+\b", sentence.lower()))
            if not sentence_words:
                continue

            common_words = entity_words & sentence_words
            # Calculate Jaccard similarity
            similarity = len(common_words) / (
                len(entity_words) + len(sentence_words) - len(common_words)
            )

            # Prioritize sentences with higher word overlap
            overlap_ratio = len(common_words) / len(entity_words) if entity_words else 0

            # Combined score giving more weight to overlap ratio
            score = (0.7 * overlap_ratio) + (0.3 * similarity)

            if score > best_score:
                best_score = score
                best_match_index = i

        # If we found a reasonably good match
        if best_score > 0.3 and best_match_index >= 0:
            return self._get_context_window(sentences, best_match_index, window_size)

        return None

    def _get_context_window(
        self, sentences: List[str], center_index: int, window_size: int
    ) -> str:
        """
        Extract a context window around the center sentence.

        Args:
            sentences: List of all sentences
            center_index: Index of the central sentence containing the entity
            window_size: Number of additional sentences to include on each side

        Returns:
            Combined context string with the specified window
        """
        # Calculate start and end indices, ensuring they stay within bounds
        start_index = max(0, center_index - window_size)
        end_index = min(len(sentences) - 1, center_index + window_size)

        # Extract the window of sentences
        context_sentences = sentences[start_index : end_index + 1]

        # Join sentences with space to form the context window
        return " ".join(context_sentences).strip()

    def extract_contexts(
        self, entities: List[str], text: str, window_size: int = 0
    ) -> List[Dict[str, str]]:
        """
        Extract contexts for a list of entities from a clinical note.

        Args:
            entities: List of entities to find context for
            text: Clinical note text
            window_size: Number of additional sentences to include (default: 0 - just the matching sentence)

        Returns:
            List of dictionaries with entity and context pairs
        """
        self._debug_print(f"Extracting contexts for {len(entities)} entities")

        # Extract sentences once
        sentences = self.extract_sentences(text)

        # Find context for each entity
        results = []
        for entity in entities:
            context = self.find_entity_context(entity, sentences, window_size)
            results.append(
                {
                    "entity": entity,
                    "context": context or "",  # Empty string if no context found
                }
            )

            self._debug_print(f"Entity: '{entity}'", level=1)
            self._debug_print(f"Context: '{context}'", level=2)

        return results

    def batch_extract_contexts(
        self, batch_entities: List[List[str]], texts: List[str], window_size: int = 0
    ) -> List[List[Dict[str, str]]]:
        """
        Extract contexts for multiple batches of entities from multiple clinical notes.

        Args:
            batch_entities: List of lists of entities
            texts: List of clinical note texts
            window_size: Number of additional sentences to include (default: 0 - just the matching sentence)

        Returns:
            List of lists of dictionaries with entity and context pairs
        """
        if len(batch_entities) != len(texts):
            raise ValueError(
                f"Mismatch between number of entity batches ({len(batch_entities)}) and texts ({len(texts)})"
            )

        results = []
        for entities, text in zip(batch_entities, texts):
            batch_results = self.extract_contexts(entities, text, window_size)
            results.append(batch_results)

        return results


# Standalone function for simpler usage
def extract_entity_contexts(
    entities: List[str], text: str, window_size: int = 0
) -> List[Dict[str, str]]:
    """
    Utility function to extract contexts for entities from a clinical note.

    Args:
        entities: List of entities to find context for
        text: Clinical note text
        window_size: Number of additional sentences to include (default: 0 - just the matching sentence)

    Returns:
        List of dictionaries with entity and context pairs
    """
    extractor = ContextExtractor()
    return extractor.extract_contexts(entities, text, window_size)
