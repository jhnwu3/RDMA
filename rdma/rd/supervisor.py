"""
RDMA Supervisor module for supervising and refining rare disease matching results.

This module provides the RDMASupervisor class which analyses and refines the
results of the rare disease matching process by examining potential false positives
and false negatives, providing human reviewable insights.

Author: Claude
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import traceback

from rdma.utils.embedding import EmbeddingsManager
from rdma.utils.llm_client import LocalLLMClient
from rdrag.verify import MultiStageRDVerifier


class RDMASupervisor:
    """
    Rare Disease supervision component that analyzes and refines rare disease matching results.

    Features:
    - Analyzes and categorizes false positives and false negatives
    - Provides detailed explanation for flagged entities
    - Supports verification of entities with their contexts
    - Generates human-readable reports for review
    """

    def __init__(
        self,
        llm_client: LocalLLMClient,
        embedding_manager: EmbeddingsManager,
        embedded_documents: List[Dict],
        system_prompt: str = "You are a medical expert specializing in rare diseases with extensive knowledge of Orphanet classifications.",
        abbreviations_file: Optional[str] = None,
        use_abbreviations: bool = True,
        top_k: int = 5,
        debug: bool = False,
    ):
        """
        Initialize the RDMA Supervisor.

        Args:
            llm_client: Client for LLM API calls
            embedding_manager: Manager for embeddings
            embedded_documents: Embedded documents containing ORPHA code metadata
            system_prompt: System prompt for LLM
            abbreviations_file: Path to abbreviations embeddings file (optional)
            use_abbreviations: Whether to use abbreviation resolution
            top_k: Number of top candidates to include in verification
            debug: Whether to print debug information
        """
        self.llm_client = llm_client
        self.embedding_manager = embedding_manager
        self.embedded_documents = embedded_documents
        self.system_prompt = system_prompt
        self.abbreviations_file = abbreviations_file
        self.use_abbreviations = use_abbreviations
        self.top_k = top_k
        self.debug = debug

        # Initialize the verifier
        self._initialize_verifier()

    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")

    def _initialize_verifier(self):
        """Initialize the verifier for entity verification."""
        self._debug_print("Initializing MultiStageRDVerifier")
        self.verifier = MultiStageRDVerifier(
            embedding_manager=self.embedding_manager,
            llm_client=self.llm_client,
            config=None,  # No specific config needed
            debug=self.debug,
            abbreviations_file=self.abbreviations_file,
            use_abbreviations=self.use_abbreviations,
        )

        # Prepare verifier index
        self._debug_print("Preparing verifier index")
        self.verifier.prepare_index(self.embedded_documents)

    def extract_entities_with_context(
        self, predictions_data: Dict, ground_truth_data: Dict, evaluation_data: Dict
    ) -> Dict[str, List[Dict]]:
        """
        Extract entities and their contexts categorized as false negatives, false positives, and true positives.

        Args:
            predictions_data: Dictionary containing prediction results
            ground_truth_data: Dictionary containing ground truth data
            evaluation_data: Dictionary containing evaluation results

        Returns:
            Dictionary with categorized entities and their contexts
        """
        # Initialize containers for categorized entities
        entities = {"false_negatives": [], "false_positives": [], "true_positives": []}

        # Extract false negatives from evaluation results
        if "corpus_false_negatives" in evaluation_data:
            for fn in evaluation_data["corpus_false_negatives"]:
                sample_id = fn.get("sample_id")
                code = fn.get("code")

                if not sample_id or not code:
                    continue

                # Find corresponding ground truth entry
                gt_entity = None
                gt_context = None

                # Look up the sample in ground truth
                if sample_id in ground_truth_data:
                    gt_sample = ground_truth_data[sample_id]

                    # Extract from annotations
                    if isinstance(gt_sample, dict) and "annotations" in gt_sample:
                        for ann in gt_sample["annotations"]:
                            ordo_field = ann.get("ordo_with_desc", "")

                            # Check if this annotation matches our code
                            if ordo_field.startswith(code) or code in ordo_field:
                                gt_entity = ann.get("mention", "")
                                # Get context if available
                                document_text = gt_sample.get("note_details", {}).get(
                                    "text", ""
                                )

                                # Extract context around the entity if possible
                                if (
                                    document_text
                                    and gt_entity
                                    and gt_entity in document_text
                                ):
                                    # Extract a simple context window
                                    pos = document_text.find(gt_entity)
                                    start = max(0, pos - 100)
                                    end = min(
                                        len(document_text), pos + len(gt_entity) + 100
                                    )
                                    gt_context = document_text[start:end]
                                break

                # Only include entities with context
                if gt_entity and gt_context:
                    entities["false_negatives"].append(
                        {
                            "entity": gt_entity,
                            "context": gt_context,
                            "orpha_code": code,
                            "document_id": sample_id,
                        }
                    )

        # Extract false positives and true positives from evaluation results
        for category, eval_key in [
            ("false_positives", "corpus_false_positives"),
            ("true_positives", "corpus_true_positives"),
        ]:
            if eval_key in evaluation_data:
                for item in evaluation_data[eval_key]:
                    sample_id = item.get("sample_id")
                    code = item.get("code")

                    if not sample_id or not code:
                        continue

                    # Find corresponding prediction
                    pred_entity = None
                    pred_context = None
                    top_candidates = None  # Store top_candidates from matched entity
                    expanded_term = None  # Store expanded abbreviation if available

                    # Look up sample in predictions
                    if sample_id in predictions_data:
                        pred_sample = predictions_data[sample_id]
                        clinical_text = pred_sample.get("clinical_text", "")

                        # Look for the entity in matched_diseases
                        if "matched_diseases" in pred_sample:
                            for match in pred_sample["matched_diseases"]:
                                # Check if this match corresponds to our code
                                orpha_id = match.get("orpha_id", "")
                                if orpha_id == code or code in orpha_id:
                                    pred_entity = match.get("entity", "")

                                    # Preserve expanded term if available
                                    if "expanded_term" in match:
                                        expanded_term = match["expanded_term"]

                                    # Preserve top_candidates if available
                                    if "top_candidates" in match:
                                        top_candidates = match["top_candidates"]

                                    # Extract context from the clinical text
                                    if (
                                        clinical_text
                                        and pred_entity
                                        and pred_entity in clinical_text
                                    ):
                                        pos = clinical_text.find(pred_entity)
                                        start = max(0, pos - 100)
                                        end = min(
                                            len(clinical_text),
                                            pos + len(pred_entity) + 100,
                                        )
                                        pred_context = clinical_text[start:end]
                                    # If entity not found in clinical text but context is available in match
                                    elif "context" in match:
                                        pred_context = match["context"]
                                    break

                    # Only include entities with context
                    if pred_entity and pred_context:
                        entity_data = {
                            "entity": pred_entity,
                            "context": pred_context,
                            "orpha_code": code,
                            "document_id": sample_id,
                        }

                        # Add expanded term if available
                        if expanded_term:
                            entity_data["expanded_term"] = expanded_term

                        # Add top_candidates if available
                        if top_candidates:
                            entity_data["top_candidates"] = top_candidates

                        entities[category].append(entity_data)

        # Print statistics
        if self.debug:
            self._debug_print(f"Extracted entities with context:")
            self._debug_print(f"  False Negatives: {len(entities['false_negatives'])}")
            self._debug_print(f"  False Positives: {len(entities['false_positives'])}")
            self._debug_print(f"  True Positives: {len(entities['true_positives'])}")

        return entities

    def process_entities(
        self, entities: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Process and verify all categorized entities using the MultiStageRDVerifier.

        Args:
            entities: Dictionary with categorized entities

        Returns:
            Dictionary with verification results for each category
        """
        results = {"false_negatives": [], "false_positives": [], "true_positives": []}

        # Process each category
        for category in ["false_negatives", "false_positives", "true_positives"]:
            category_entities = entities[category]

            self._debug_print(f"Processing {len(category_entities)} {category}...")

            # Process each entity
            for entity_data in category_entities:
                try:
                    entity = entity_data["entity"]
                    context = entity_data["context"]
                    orpha_code = entity_data["orpha_code"]
                    document_id = entity_data["document_id"]

                    if self.debug:
                        self._debug_print(
                            f"Processing {category} entity: '{entity}' (Document: {document_id})",
                            level=1,
                        )

                    # Handle entity edge cases
                    if entity is None and "original_entity" in entity_data:
                        entity = entity_data["original_entity"]

                    # Process the entity through MultiStageRDVerifier
                    verification_result = self.verifier.process_entity(entity, context)

                    # Check if it's an abbreviation
                    is_abbreviation = False
                    expanded_term = None

                    if "expanded_term" in verification_result:
                        is_abbreviation = True
                        expanded_term = verification_result["expanded_term"]
                        if self.debug:
                            self._debug_print(
                                f"  Detected abbreviation: '{entity}' expands to '{expanded_term}'",
                                level=1,
                            )

                    # Determine if we should flag for review based on category and verification result
                    flag_for_review = False
                    explanation = verification_result.get("method", "")

                    # Determine if the entity is a rare disease according to the verifier
                    is_rare_disease = (
                        verification_result.get("status") == "verified_rare_disease"
                    )

                    # Logic for flagging based on category and verification result
                    if category == "false_positives" and is_rare_disease:
                        # Contradiction: Verifier thinks it's a rare disease but it's marked as false positive
                        flag_for_review = True
                        explanation += " [FLAGGED: Entity verified as a rare disease despite being categorized as false positive]"
                    elif category == "false_negatives" and not is_rare_disease:
                        # Contradiction: Verifier thinks it's not a rare disease but it's marked as false negative
                        flag_for_review = True
                        explanation += " [FLAGGED: Entity verified as not a rare disease despite being categorized as false negative]"
                    elif category == "true_positives" and not is_rare_disease:
                        # Contradiction: Verifier thinks it's not a rare disease but it's marked as true positive
                        flag_for_review = True
                        explanation += " [FLAGGED: Entity verified as not a rare disease despite being categorized as true positive]"

                    # Create the final result
                    result = {
                        "entity": entity,
                        "context": context,
                        "is_rare_disease": is_rare_disease,
                        "flag_for_review": flag_for_review,
                        "explanation": explanation,
                        "category": category,
                        "document_id": document_id,
                        "orpha_code": orpha_code,
                        "verification_method": verification_result.get("method", ""),
                    }

                    # Add expanded term if it's an abbreviation
                    if is_abbreviation and expanded_term:
                        result["expanded_term"] = expanded_term

                    # Add candidates if available (from top_candidates or from new retrieval)
                    if (
                        "top_candidates" in entity_data
                        and entity_data["top_candidates"]
                    ):
                        result["orpha_candidates"] = entity_data["top_candidates"]
                    else:
                        # Use query for retrieval based on expanded term if available
                        query_term = expanded_term if expanded_term else entity
                        candidates = self.verifier._retrieve_similar_diseases(
                            query_term
                        )
                        if candidates:
                            # The candidates from _retrieve_similar_diseases have a different structure
                            result["orpha_candidates"] = [
                                {
                                    "name": c.get("name", ""),
                                    "id": c.get("id", ""),
                                    "similarity": float(c.get("similarity_score", 0.0)),
                                }
                                for c in candidates[
                                    : self.top_k
                                ]  # Include up to top_k candidates
                            ]

                    # Add to results
                    results[category].append(result)

                    # Debug output
                    if self.debug:
                        self._debug_print(
                            f"  Result: is_rare_disease={result['is_rare_disease']}, "
                            f"flag_for_review={result['flag_for_review']}",
                            level=1,
                        )
                        self._debug_print(
                            f"  Explanation: {result['explanation']}", level=1
                        )

                except Exception as e:
                    self._debug_print(
                        f"Error processing {category} entity '{entity_data.get('entity', '')}': {str(e)}",
                        level=1,
                    )
                    if self.debug:
                        traceback.print_exc()

                    # Add error result
                    error_result = {
                        "entity": entity_data.get("entity", ""),
                        "context": entity_data.get("context", ""),
                        "document_id": entity_data.get("document_id", ""),
                        "orpha_code": entity_data.get("orpha_code", ""),
                        "is_rare_disease": False,
                        "flag_for_review": True,  # Always flag for review on errors
                        "explanation": f"Error during processing: {str(e)}",
                        "category": category,
                        "error": str(e),
                    }

                    results[category].append(error_result)

        return results

    def prepare_summary(
        self, verification_results: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Prepare a summary of the verification results.

        Args:
            verification_results: Dictionary with verification results for each category

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "categories": {},
        }

        # Initialize arrays for overall statistics
        all_flagged_for_review = []

        # Calculate statistics for each category
        for category in ["false_negatives", "false_positives", "true_positives"]:
            results = verification_results[category]

            # Initialize category statistics
            category_stats = {
                "total": len(results),
                "confirmed_rare_disease_count": 0,
                "confirmed_rare_disease_percentage": 0.0,
                "flagged_for_review_count": 0,
                "flagged_for_review_percentage": 0.0,
                "confirmation_status": {},
            }

            # Count by confirmation and review status
            is_rare_counts = {"YES": 0, "NO": 0}
            flag_for_review_counts = {"YES": 0, "NO": 0}

            for result in results:
                # Count by rare disease status
                if result.get("is_rare_disease", False):
                    is_rare_counts["YES"] += 1
                else:
                    is_rare_counts["NO"] += 1

                # Count by review flag status
                if result.get("flag_for_review", False):
                    flag_for_review_counts["YES"] += 1
                    # Add to overall flagged list
                    all_flagged_for_review.append(
                        {
                            "entity": result.get("entity", ""),
                            "document_id": result.get("document_id", ""),
                            "orpha_code": result.get("orpha_code", ""),
                            "category": category,
                            "explanation": result.get("explanation", ""),
                        }
                    )
                else:
                    flag_for_review_counts["NO"] += 1

            # Calculate percentages
            if category_stats["total"] > 0:
                category_stats["confirmed_rare_disease_count"] = is_rare_counts["YES"]
                category_stats["confirmed_rare_disease_percentage"] = (
                    is_rare_counts["YES"] / category_stats["total"] * 100
                )

                category_stats["flagged_for_review_count"] = flag_for_review_counts[
                    "YES"
                ]
                category_stats["flagged_for_review_percentage"] = (
                    flag_for_review_counts["YES"] / category_stats["total"] * 100
                )

            # Store detailed status counts
            category_stats["confirmation_status"] = {
                "is_rare_disease": is_rare_counts,
                "flag_for_review": flag_for_review_counts,
            }

            # Add to summary
            summary["categories"][category] = category_stats

        # Add overall statistics
        summary["total_entities"] = sum(
            summary["categories"][cat]["total"] for cat in summary["categories"]
        )
        summary["total_flagged_for_review"] = len(all_flagged_for_review)
        summary["flagged_for_review_percentage"] = (
            (summary["total_flagged_for_review"] / summary["total_entities"] * 100)
            if summary["total_entities"] > 0
            else 0.0
        )

        # Add flagged entities list for easy access
        summary["flagged_entities"] = all_flagged_for_review

        return summary

    def supervise(
        self,
        predictions_data: Dict[str, Dict],
        ground_truth_data: Dict[str, Dict],
        evaluation_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Supervise the rare disease matching process by analyzing false positives and negatives.

        Args:
            predictions_data: Dictionary containing prediction results
            ground_truth_data: Dictionary containing ground truth data
            evaluation_data: Dictionary containing evaluation results

        Returns:
            Dictionary with supervision results
        """
        # Extract entities with contexts
        self._debug_print("Extracting entities with contexts")
        entities = self.extract_entities_with_context(
            predictions_data, ground_truth_data, evaluation_data
        )

        # Process entities with the verifier
        self._debug_print("Starting entity verification")
        verification_results = self.process_entities(entities)

        # Prepare summary
        self._debug_print("Preparing results summary")
        summary = self.prepare_summary(verification_results)

        # Create final result with all components
        result = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "predictions_file": getattr(predictions_data, "source", "unknown"),
                "ground_truth_file": getattr(ground_truth_data, "source", "unknown"),
                "evaluation_file": getattr(evaluation_data, "source", "unknown"),
                "model_info": {
                    "system_prompt": self.system_prompt,
                    "use_abbreviations": self.use_abbreviations,
                },
            },
            "summary": summary,
            "results": verification_results,
        }

        return result

    def save_results(self, supervision_results: Dict[str, Any], output_file: str):
        """
        Save supervision results to a JSON file.

        Args:
            supervision_results: Results dictionary to save
            output_file: File path to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Save to file
        with open(output_file, "w") as f:
            json.dump(supervision_results, f, indent=2)

        if self.debug:
            self._debug_print(f"Results saved to {output_file}")

    def save_flagged_entities_csv(
        self, supervision_results: Dict[str, Any], csv_file: str
    ):
        """
        Save flagged entities to a CSV file for human review.

        Args:
            supervision_results: Dictionary with supervision results
            csv_file: File path to save CSV
        """
        # Extract flagged entities
        flagged_entities = supervision_results.get("summary", {}).get(
            "flagged_entities", []
        )

        if not flagged_entities:
            self._debug_print("No flagged entities to save")
            return

        # Convert to DataFrame
        df = pd.DataFrame(flagged_entities)

        # Save to CSV
        df.to_csv(csv_file, index=False)

        if self.debug:
            self._debug_print(
                f"Saved {len(flagged_entities)} flagged entities to {csv_file}"
            )

    def print_supervision_summary(self, supervision_results: Dict[str, Any]):
        """
        Print a summary of the supervision results.

        Args:
            supervision_results: Dictionary with supervision results
        """
        summary = supervision_results.get("summary", {})

        print("\n=== Supervision Summary ===")
        print(f"Total entities processed: {summary.get('total_entities', 0)}")
        print(
            f"Total entities flagged for review: {summary.get('total_flagged_for_review', 0)} "
            f"({summary.get('flagged_for_review_percentage', 0):.1f}%)"
        )

        for category in ["false_negatives", "false_positives", "true_positives"]:
            stats = summary.get("categories", {}).get(category, {})
            print(f"\n{category.replace('_', ' ').title()}:")
            print(f"  Total: {stats.get('total', 0)}")
            print(
                f"  Confirmed as rare disease: {stats.get('confirmed_rare_disease_count', 0)} "
                f"({stats.get('confirmed_rare_disease_percentage', 0):.1f}%)"
            )
            print(
                f"  Flagged for review: {stats.get('flagged_for_review_count', 0)} "
                f"({stats.get('flagged_for_review_percentage', 0):.1f}%)"
            )
