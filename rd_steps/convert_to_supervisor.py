import json
import os
import numpy as np
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Set
import traceback


# Helper class for handling NumPy types in JSON serialization
class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def find_all_occurrences(entity: str, text: str) -> List[Tuple[int, int]]:
    """
    Find all occurrences of an entity string in the text (exact or case-insensitive).
    Fuzzy matching is removed for simplicity and performance.

    Args:
        entity: Entity string to find
        text: Text to search in

    Returns:
        List of tuples (start, end) for each occurrence, sorted by start position.
    """
    occurrences = []

    if not entity or not text:
        # print(f"DEBUG: find_all_occurrences called with empty entity or text.")
        return occurrences

    # print(f"DEBUG: Searching for '{entity}' in text...")

    # 1. Look for exact matches (case-sensitive)
    start_idx = 0
    while True:
        pos = text.find(entity, start_idx)
        if pos == -1:
            break
        occurrences.append((pos, pos + len(entity)))
        start_idx = pos + 1  # Move past this occurrence

    # 2. Look for case-insensitive matches and add any new ones
    text_lower = text.lower()
    entity_lower = entity.lower()
    start_idx = 0
    while True:
        pos = text_lower.find(entity_lower, start_idx)
        if pos == -1:
            break
        # Check if this position is already covered by an exact match
        # This de-duplicates positions found by both methods
        if (pos, pos + len(entity)) not in occurrences:
            occurrences.append((pos, pos + len(entity)))
        start_idx = pos + 1

    # Sort by position
    occurrences.sort()

    # print(f"DEBUG: Found {len(occurrences)} occurrences for '{entity}' at positions: {occurrences}")

    return occurrences


def extract_context_for_entities(entity_list, clinical_text, window_size=100):
    """
    Extract contexts for a list of entities that may contain duplicates.

    Args:
        entity_list: List of entity strings, potentially with duplicates
        clinical_text: Full clinical text to search in
        window_size: Size of context window around entity

    Returns:
        List of (entity, context, (start_pos, end_pos)) tuples
    """
    # First, find ALL occurrences of ALL entities
    all_entity_occurrences = {}
    for entity in set(entity_list):  # Process unique entities first
        occurrences = find_all_occurrences(entity, clinical_text)
        all_entity_occurrences[entity] = occurrences

    # Count occurrences of each entity in the original list to preserve order
    entity_counts = {}
    for entity in entity_list:
        entity_counts[entity] = entity_counts.get(entity, 0) + 1

    # Track used positions to avoid overlaps
    used_positions = set()

    # Results list with same order as input list
    results = []

    # Process each entity instance in the original order
    entity_instance_tracker = {}  # Track which occurrence to use for each entity

    for entity in entity_list:
        # Get occurrence index for this instance of the entity
        occurrence_idx = entity_instance_tracker.get(entity, 0)
        entity_instance_tracker[entity] = occurrence_idx + 1

        # Get all possible occurrences for this entity
        all_occurrences = all_entity_occurrences.get(entity, [])

        # Find a valid, non-overlapping occurrence
        context = ""
        position = (-1, -1)

        if occurrence_idx < len(all_occurrences):
            # Try to use the corresponding occurrence
            candidate_position = all_occurrences[occurrence_idx]

            # Check if this position overlaps with previously used positions
            overlaps = False
            for used_start, used_end in used_positions:
                if (
                    candidate_position[0] < used_end
                    and candidate_position[1] > used_start
                ):
                    overlaps = True
                    break

            if not overlaps:
                # Position is valid, extract context
                start_pos, end_pos = candidate_position
                start_context = max(0, start_pos - window_size)
                end_context = min(len(clinical_text), end_pos + window_size)
                context = clinical_text[start_context:end_context]
                position = (start_pos, end_pos)

                # Mark position as used
                used_positions.add(position)

        # Add result even if no valid context was found
        if position[0] == -1:
            context = f"[Entity '{entity}' occurrence #{occurrence_idx + 1} not found or overlaps]"

        results.append((entity, context, position))

    return results


# Rest of the convert_to_supervisor_format function remains largely the same,
# but let's ensure the skipped entity/document handling is clear and counters are correct.


def extract_context_from_annotation_span(
    start_pos: int,
    end_pos: int,
    clinical_text: str,
    used_positions: Set[Tuple[int, int]],
    window_size: int = 100,
) -> Tuple[str, Tuple[int, int]]:
    """
    Extract context around a specific span provided by an annotation.
    Checks for overlap with previously used spans. Does NOT add brackets.

    Args:
        start_pos: Start position of the annotation span
        end_pos: End position of the annotation span
        clinical_text: Full clinical text
        used_positions: Set of already used positions (start, end) (mutated)
        window_size: Size of context window around entity

    Returns:
        Tuple of (context_string, (start_pos, end_pos)).
        Returns ("", (-1, -1)) if the requested span is invalid or overlaps.
    """
    if start_pos < 0 or end_pos <= start_pos or end_pos > len(clinical_text):
        print(f"DEBUG: Invalid span requested: ({start_pos}, {end_pos})")
        return "", (-1, -1)

    # Check for overlap with previously used positions
    overlaps = False
    for used_start, used_end in used_positions:
        if start_pos < used_end and end_pos > used_start:
            overlaps = True
            print(
                f"DEBUG: Span ({start_pos}, {end_pos}) overlaps with used position ({used_start}, {used_end}). Skipping."
            )
            break

    if overlaps:
        return "", (-1, -1)

    # Extract context
    start_context = max(0, start_pos - window_size)
    end_context = min(len(clinical_text), end_pos + window_size)
    context = clinical_text[start_context:end_context]

    # Add the found position to the set of used positions for *this document*
    used_positions.add((start_pos, end_pos))
    print(f"DEBUG: Added ({start_pos}, {end_pos}) to used_positions.")

    # Return the context and the exact position
    return context, (start_pos, end_pos)


def convert_to_supervisor_format(input_data: Dict, window_size: int = 100) -> Dict:
    """
    Convert dataset to the format matching step4_supervisor output.

    This function takes a dataset with clinical notes and annotations and converts it
    to the format expected by the step4_supervisor.py script. It extracts contexts for
    each annotation based on its occurrence in the text and flags them all for review
    as 'false_negatives'. Contexts do not include extra brackets.

    Args:
        input_data: Dictionary with document_id -> document data mapping
        window_size: Size of context window around entities (default: 100 chars)

    Returns:
        Dictionary formatted like step4_supervisor output
    """
    # Initialize results structure (copying the original structure)
    results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predictions_file": "NA",
            "ground_truth_file": "manual_conversion",
            "evaluation_file": "NA",
            "model_info": {
                "system_prompt": "Manual conversion to supervisor format",
            },
        },
        "summary": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_entities": 0,
            "total_flagged_for_review": 0,
            "flagged_for_review_percentage": 0.0,
            "categories": {
                "false_negatives": {
                    "total": 0,
                    "confirmed_rare_disease_count": 0,
                    "confirmed_rare_disease_percentage": 0.0,
                    "flagged_for_review_count": 0,
                    "flagged_for_review_percentage": 0.0,
                    "confirmation_status": {
                        "is_rare_disease": {"YES": 0, "NO": 0},
                        "flag_for_review": {"YES": 0, "NO": 0},
                    },
                },
                # Initialize other categories even if not used in this conversion
                "false_positives": {
                    "total": 0,
                    "confirmed_rare_disease_count": 0,
                    "confirmed_rare_disease_percentage": 0.0,
                    "flagged_for_review_count": 0,
                    "flagged_for_review_percentage": 0.0,
                    "confirmation_status": {
                        "is_rare_disease": {"YES": 0, "NO": 0},
                        "flag_for_review": {"YES": 0, "NO": 0},
                    },
                },
                "true_positives": {
                    "total": 0,
                    "confirmed_rare_disease_count": 0,
                    "confirmed_rare_disease_percentage": 0.0,
                    "flagged_for_review_count": 0,
                    "flagged_for_review_percentage": 0.0,
                    "confirmation_status": {
                        "is_rare_disease": {"YES": 0, "NO": 0},
                        "flag_for_review": {"YES": 0, "NO": 0},
                    },
                },
            },
            "flagged_entities": [],
        },
        "results": {
            "false_negatives": [],
            "false_positives": [],  # Initialize empty lists for other categories
            "true_positives": [],
        },
    }

    # Statistics for reporting
    entity_found_in_context_count = 0
    entity_not_found_in_context_count = 0
    annotation_processed_count = 0
    annotation_skipped_count = 0
    doc_processed_successfully_count = 0
    doc_skipped_count = 0
    entity_found_by_span_count = 0
    entity_found_by_string_count = 0

    # Process each document
    for doc_id, doc_data in input_data.items():
        clinical_text = None
        annotations = []
        process_doc = True

        try:
            # Extract clinical text and annotations
            if isinstance(doc_data, dict):
                if "note_details" in doc_data and isinstance(
                    doc_data["note_details"], dict
                ):
                    clinical_text = doc_data["note_details"].get("text", "")
                    annotations = doc_data.get("annotations", []) + doc_data[
                        "note_details"
                    ].get("annotations", [])
                if not clinical_text:
                    clinical_text = doc_data.get("clinical_text", "") or doc_data.get(
                        "text", ""
                    )
                if not annotations:
                    annotations = doc_data.get("annotations", []) or doc_data.get(
                        "gold_annotations", []
                    )
            elif isinstance(doc_data, str):
                clinical_text = doc_data
                annotations = []  # No annotations in this simple format

            if not clinical_text:
                print(
                    f"Warning: No clinical text found for document {doc_id}. Skipping document."
                )
                doc_skipped_count += 1
                process_doc = False
            if not annotations and clinical_text:
                print(
                    f"Warning: No annotations found for document {doc_id} despite having text."
                )
                process_doc = False

            if process_doc:
                doc_processed_successfully_count += 1
                # Track used positions in the text for *this document* to avoid overlapping contexts
                used_positions: Set[Tuple[int, int]] = set()

                # REVISED APPROACH: Pre-compute all entity occurrences and track their instance counts
                all_entity_occurrences = {}
                annotation_entities = []
                span_annotations = []

                # First pass: collect all entities and their info
                for annotation in annotations:
                    entity = annotation.get("mention", "")
                    if not entity:
                        print(
                            f"Warning: Empty entity mention in annotation in document {doc_id}. Skipping annotation."
                        )
                        annotation_skipped_count += 1
                        continue

                    # Add to our tracking lists
                    annotation_entities.append(entity)
                    span_annotations.append(annotation)

                # Find all occurrences of each unique entity in the text
                for entity in set(annotation_entities):
                    all_entity_occurrences[entity] = find_all_occurrences(
                        entity, clinical_text
                    )

                # Track instance number for each entity (which occurrence we're up to for each entity name)
                entity_instance_tracker = {}

                # Process each annotation with the correct occurrence index
                for i, (annotation, entity) in enumerate(
                    zip(span_annotations, annotation_entities)
                ):
                    annotation_processed_count += 1

                    # Get ORPHA ID
                    orpha_id = ""
                    if "ordo_with_desc" in annotation and isinstance(
                        annotation["ordo_with_desc"], str
                    ):
                        ordo_parts = annotation["ordo_with_desc"].split(maxsplit=1)
                        if ordo_parts:
                            orpha_id = ordo_parts[0]
                    elif "orpha_id" in annotation:
                        orpha_id = str(annotation["orpha_id"])
                    elif "concept_id" in annotation:
                        orpha_id = str(annotation["concept_id"])

                    # Try with annotation spans first if available
                    annotated_start = annotation.get("start")
                    annotated_end = annotation.get("end")

                    context = ""
                    start_pos = -1
                    end_pos = -1
                    found_by_span = False

                    # 1. Try to use annotation start/end if available and valid
                    if (
                        isinstance(annotated_start, int)
                        and isinstance(annotated_end, int)
                        and annotated_start >= 0
                        and annotated_end > annotated_start
                    ):

                        print(
                            f"\nDEBUG: Processing annotation '{entity}' using span ({annotated_start}, {annotated_end})"
                        )

                        # Check for overlap
                        overlaps = False
                        for used_start, used_end in used_positions:
                            if (
                                annotated_start < used_end
                                and annotated_end > used_start
                            ):
                                overlaps = True
                                print(
                                    f"DEBUG: Span ({annotated_start}, {annotated_end}) overlaps with used position ({used_start}, {used_end}). Skipping."
                                )
                                break

                        if not overlaps:
                            # Extract context using span
                            start_context = max(0, annotated_start - window_size)
                            end_context = min(
                                len(clinical_text), annotated_end + window_size
                            )
                            context = clinical_text[start_context:end_context]
                            start_pos = annotated_start
                            end_pos = annotated_end

                            # Mark position as used
                            used_positions.add((start_pos, end_pos))

                            found_by_span = True
                            entity_found_by_span_count += 1
                            print(f"DEBUG: Found by span: ({start_pos}, {end_pos})")

                    # 2. If span failed or wasn't available, use string search with proper occurrence tracking
                    if not found_by_span:
                        # Track which occurrence number this is for debugging purposes
                        occurrence_idx = entity_instance_tracker.get(entity, 0)
                        entity_instance_tracker[entity] = occurrence_idx + 1

                        print(
                            f"\nDEBUG: Processing annotation '{entity}' using string search (occurrence #{occurrence_idx + 1})"
                        )

                        # Get all occurrences for this entity
                        occurrences = all_entity_occurrences.get(entity, [])

                        # Filter out overlapping occurrences
                        valid_occurrences = []
                        for occ_start, occ_end in occurrences:
                            overlaps = False
                            for used_start, used_end in used_positions:
                                if occ_start < used_end and occ_end > used_start:
                                    overlaps = True
                                    print(
                                        f"DEBUG: Occurrence ({occ_start}, {occ_end}) overlaps with used position ({used_start}, {used_end}). Skipping."
                                    )
                                    break

                            if not overlaps:
                                valid_occurrences.append((occ_start, occ_end))

                        print(
                            f"DEBUG: Valid (non-overlapping) occurrences: {valid_occurrences}"
                        )

                        # If we have any valid occurrences, take the first one
                        if valid_occurrences:
                            start_pos, end_pos = valid_occurrences[
                                0
                            ]  # Always use index 0

                            # Extract context
                            start_context = max(0, start_pos - window_size)
                            end_context = min(len(clinical_text), end_pos + window_size)
                            context = clinical_text[start_context:end_context]

                            # Mark position as used
                            used_positions.add((start_pos, end_pos))

                            entity_found_by_string_count += 1
                            print(
                                f"DEBUG: Found by string search: ({start_pos}, {end_pos}) (Occurrence #{occurrence_idx + 1})"
                            )
                        else:
                            print(
                                f"DEBUG: No valid non-overlapping occurrences found for '{entity}'"
                            )

                    # Determine if a context/position was successfully found by either method
                    found_in_context = start_pos >= 0 and end_pos > start_pos

                    if found_in_context:
                        entity_found_in_context_count += 1
                    else:
                        entity_not_found_in_context_count += 1
                        # Set a default context message if not found by either method
                        if found_by_span:  # Attempted by span but failed due to overlap
                            context = f"[Annotation span ({annotated_start}, {annotated_end}) for '{entity}' overlaps with a previously used position in document {doc_id} ORPHA:{orpha_id}]"
                        else:  # Fell back to string search, and it failed
                            occurrence_idx = entity_instance_tracker.get(entity, 1) - 1
                            context = f"[Entity '{entity}' occurrence #{occurrence_idx + 1} (index {occurrence_idx}) not found by string search or overlaps with previously used positions in document {doc_id} ORPHA:{orpha_id}]"

                    # Create entity record for results
                    # Create entity record for results
                    print(
                        f"DEBUG: Adding entity record for '{entity}' with context: {context}"
                    )
                    entity_record = {
                        "entity": entity,
                        "context": context,  # This should be the unique context for this occurrence
                        "is_rare_disease": True,
                        "flag_for_review": True,
                        "explanation": "Original annotation requires human review"
                        + (
                            " (Found by span)"
                            if found_by_span
                            else (
                                " (Found by string search)"
                                if found_in_context
                                else " (Not found in text)"
                            )
                        ),
                        "category": "false_negatives",
                        "document_id": doc_id,
                        "orpha_code": orpha_id,
                        "verification_method": "direct_annotation"
                        + ("_span" if found_by_span else "_string"),
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                    }

                    # Do a final check to ensure we're not adding a duplicate context
                    existing_positions = [
                        (rec["start_pos"], rec["end_pos"])
                        for rec in results["results"]["false_negatives"]
                        if rec["entity"] == entity and rec["document_id"] == doc_id
                    ]
                    if (start_pos, end_pos) in existing_positions:
                        print(
                            f"WARNING: Duplicate position ({start_pos}, {end_pos}) for entity '{entity}' in document {doc_id}"
                        )

                    # Add this entity record to results
                    results["results"]["false_negatives"].append(entity_record)

                    # Create summary record for flagged entities
                    flagged_entity = {
                        "entity": entity,
                        "document_id": doc_id,
                        "orpha_code": orpha_id,
                        "category": "false_negatives",
                        "explanation": "Original annotation requires human review",
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                    }

                    # Add to results list
                    results["summary"]["flagged_entities"].append(flagged_entity)

                    # Update summary counters
                    results["summary"]["total_entities"] += 1
                    results["summary"]["total_flagged_for_review"] += 1
                    results["summary"]["categories"]["false_negatives"]["total"] += 1
                    results["summary"]["categories"]["false_negatives"][
                        "confirmed_rare_disease_count"
                    ] += 1
                    results["summary"]["categories"]["false_negatives"][
                        "flagged_for_review_count"
                    ] += 1
                    results["summary"]["categories"]["false_negatives"][
                        "confirmation_status"
                    ]["is_rare_disease"]["YES"] += 1
                    results["summary"]["categories"]["false_negatives"][
                        "confirmation_status"
                    ]["flag_for_review"]["YES"] += 1

        except Exception as e:
            print(f"Error processing document {doc_id}: {str(e)}")
            traceback.print_exc()
            if not clinical_text or not process_doc:
                pass
            else:
                doc_skipped_count += 1
            continue

    # Calculate percentages
    for category_key, category_data in results["summary"]["categories"].items():
        if category_data["total"] > 0:
            category_data["confirmed_rare_disease_percentage"] = (
                category_data["confirmed_rare_disease_count"]
                / category_data["total"]
                * 100
            )
            category_data["flagged_for_review_percentage"] = (
                category_data["flagged_for_review_count"] / category_data["total"] * 100
            )
        else:
            category_data["confirmed_rare_disease_percentage"] = 0.0
            category_data["flagged_for_review_percentage"] = 0.0

    if results["summary"]["total_entities"] > 0:
        results["summary"]["flagged_for_review_percentage"] = (
            results["summary"]["total_flagged_for_review"]
            / results["summary"]["total_entities"]
            * 100
        )
    else:
        results["summary"]["flagged_for_review_percentage"] = 0.0

    # Add processing statistics to metadata
    results["metadata"]["processing_stats"] = {
        "documents_attempted": len(input_data),
        "documents_processed_successfully": doc_processed_successfully_count,
        "documents_skipped_due_to_error_or_no_content": doc_skipped_count,
        "annotations_attempted_processing": annotation_processed_count,
        "annotations_skipped_empty_mention": annotation_skipped_count,
        "entities_added_to_results": results["summary"]["total_entities"],
        "entities_found_in_context": entity_found_in_context_count,
        "entities_found_by_span": entity_found_by_span_count,
        "entities_found_by_string_search": entity_found_by_string_count,
        "entities_not_found_in_context_or_overlapped": entity_not_found_in_context_count,
    }

    return results


def main():
    """
    Main function to run the conversion process.

    Usage:
        python convert_to_supervisor.py --input_file input.json --output_file output.json [--window_size 100]
    """
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert dataset to supervisor format for review"
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input JSON file with clinical notes and annotations",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSON file for supervisor format review",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Size of context window (default: 100 characters on each side)",
    )
    args = parser.parse_args()

    try:
        print(f"Loading input data from {args.input_file}...")
        with open(args.input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        print(f"Loaded {len(input_data)} top-level entries (potential documents)")

        print("Converting to supervisor format...")
        result = convert_to_supervisor_format(input_data, window_size=args.window_size)

        output_dir = os.path.dirname(os.path.abspath(args.output_file))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Saving result to {args.output_file}...")
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)

        stats = result["metadata"]["processing_stats"]
        summary = result["summary"]

        print("\n--- Conversion Summary ---")
        print(f"Input documents attempted: {stats['documents_attempted']}")
        print(
            f"Documents processed successfully: {stats['documents_processed_successfully']}"
        )
        print(
            f"Documents skipped (error/no content): {stats['documents_skipped_due_to_error_or_no_content']}"
        )
        print("-" * 26)
        print(
            f"Annotations attempted processing: {stats['annotations_attempted_processing']}"
        )
        print(
            f"Annotations skipped (empty mention): {stats['annotations_skipped_empty_mention']}"
        )
        print(f"Entities added to results: {stats['entities_added_to_results']}")
        print(f"  - Found and added context: {stats['entities_found_in_context']}")
        print(
            f"    - Found by annotation span: {stats['entities_found_by_span']}"
        )  # New stat
        print(
            f"    - Found by string search: {stats['entities_found_by_string_search']}"
        )  # New stat
        print(
            f"  - Not found in text or overlapped: {stats['entities_not_found_in_context_or_overlapped']}"
        )
        print("-" * 26)
        print(
            f"Total entities flagged for review: {summary['total_flagged_for_review']} ({summary['flagged_for_review_percentage']:.1f}%)"
        )
        print("All entities flagged as 'false_negatives' requiring manual review.")
        print("--------------------------")

    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        exit(1)
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from {args.input_file}. Please check file format."
        )
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
