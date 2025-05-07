import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


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


def convert_to_supervisor_format(input_data: Dict, window_size: int = 100) -> Dict:
    """
    Convert dataset to the format matching step4_supervisor output.

    This function takes a dataset with clinical notes and annotations and converts it
    to the format expected by the step4_supervisor.py script. It extracts contexts for
    each annotation and flags them all for review as 'false_negatives'.

    Args:
        input_data: Dictionary with document_id -> document data mapping
        window_size: Size of context window around entities (default: 100 chars)

    Returns:
        Dictionary formatted like step4_supervisor output
    """
    # Initialize results structure
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
        "results": {"false_negatives": [], "false_positives": [], "true_positives": []},
    }

    # Process each document
    doc_count = 0
    entity_count = 0
    skipped_docs = 0
    skipped_entities = 0

    for doc_id, doc_data in input_data.items():
        doc_count += 1

        # Extract clinical text and annotations
        clinical_text = None
        annotations = []

        try:
            # Check if we have MIMIC format with note_details
            if isinstance(doc_data, dict) and "note_details" in doc_data:
                note_details = doc_data["note_details"]
                clinical_text = note_details.get("text", "")
                if "annotations" in doc_data:
                    annotations = doc_data["annotations"]
            # Check if we have simpler format
            elif isinstance(doc_data, dict):
                clinical_text = doc_data.get("clinical_text", "") or doc_data.get(
                    "text", ""
                )
                annotations = doc_data.get("annotations", []) or doc_data.get(
                    "gold_annotations", []
                )
            # Check if it's just a string (simplest format)
            elif isinstance(doc_data, str):
                clinical_text = doc_data

            # Skip if no text
            if not clinical_text:
                print(f"Warning: No clinical text found for document {doc_id}")
                skipped_docs += 1
                continue

            # Skip if no annotations
            if not annotations:
                print(f"Warning: No annotations found for document {doc_id}")
                skipped_docs += 1
                continue

            # Process each annotation
            for annotation in annotations:
                entity_count += 1
                entity = annotation.get("mention", "")
                orpha_id = ""

                # Extract ORPHA ID if available
                if "ordo_with_desc" in annotation:
                    ordo_with_desc = annotation["ordo_with_desc"]
                    if isinstance(ordo_with_desc, str):
                        ordo_parts = ordo_with_desc.split(maxsplit=1)
                        if ordo_parts:
                            orpha_id = ordo_parts[0]
                elif "orpha_id" in annotation:
                    orpha_id = annotation["orpha_id"]

                # Skip if no entity
                if not entity:
                    print(f"Warning: Empty entity in document {doc_id}")
                    skipped_entities += 1
                    continue

                # Extract context around entity
                context = ""
                try:
                    if entity in clinical_text:
                        start_pos = clinical_text.find(entity)
                        start_context = max(0, start_pos - window_size)
                        end_context = min(
                            len(clinical_text), start_pos + len(entity) + window_size
                        )
                        context = clinical_text[start_context:end_context]
                    else:
                        print(
                            f"Warning: Entity '{entity}' not found in document {doc_id}"
                        )
                        # Use a small portion of the text as context
                        context = clinical_text[: min(200, len(clinical_text))] + "..."
                except Exception as e:
                    print(
                        f"Error extracting context for '{entity}' in document {doc_id}: {e}"
                    )
                    context = "Error extracting context"

                # Create entity record for results
                entity_record = {
                    "entity": entity,
                    "context": context,
                    "is_rare_disease": True,  # Assuming all annotations are rare diseases
                    "flag_for_review": True,  # Flag all for review as requested
                    "explanation": "Annotation requires human review",
                    "category": "false_negatives",  # All are false negatives as requested
                    "document_id": doc_id,
                    "orpha_code": orpha_id,
                    "verification_method": "direct_annotation",
                }

                # Create summary record for flagged entities
                flagged_entity = {
                    "entity": entity,
                    "document_id": doc_id,
                    "orpha_code": orpha_id,
                    "category": "false_negatives",
                    "explanation": "Annotation requires human review",
                }

                # Add to results
                results["results"]["false_negatives"].append(entity_record)

                # Add to flagged entities
                results["summary"]["flagged_entities"].append(flagged_entity)

                # Update counters
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
            print(f"Error processing document {doc_id}: {e}")
            skipped_docs += 1

    # Calculate percentages
    for category in ["false_negatives", "false_positives", "true_positives"]:
        cat_data = results["summary"]["categories"][category]
        if cat_data["total"] > 0:
            cat_data["confirmed_rare_disease_percentage"] = (
                cat_data["confirmed_rare_disease_count"] / cat_data["total"]
            ) * 100
            cat_data["flagged_for_review_percentage"] = (
                cat_data["flagged_for_review_count"] / cat_data["total"]
            ) * 100

    if results["summary"]["total_entities"] > 0:
        results["summary"]["flagged_for_review_percentage"] = (
            results["summary"]["total_flagged_for_review"]
            / results["summary"]["total_entities"]
        ) * 100

    # Add processing statistics to metadata
    results["metadata"]["processing_stats"] = {
        "documents_processed": doc_count,
        "documents_skipped": skipped_docs,
        "entities_processed": entity_count,
        "entities_skipped": skipped_entities,
        "entities_converted": results["summary"]["total_entities"],
    }

    return results


def main():
    """
    Main function to run the conversion process.

    Usage:
        python convert_to_supervisor.py --input_file input.json --output_file output.json
    """
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert dataset to supervisor format")
    parser.add_argument(
        "--input_file", required=True, help="Input JSON file with clinical notes"
    )
    parser.add_argument(
        "--output_file", required=True, help="Output JSON file for supervisor format"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Size of context window (default: 100)",
    )
    args = parser.parse_args()

    try:
        # Load input data
        print(f"Loading input data from {args.input_file}...")
        with open(args.input_file, "r") as f:
            input_data = json.load(f)

        print(f"Loaded {len(input_data)} documents")

        # Convert to supervisor format
        print("Converting to supervisor format...")
        result = convert_to_supervisor_format(input_data, window_size=args.window_size)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

        # Save result
        print(f"Saving result to {args.output_file}...")
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2, cls=NumpyJSONEncoder)

        # Print summary
        stats = result["metadata"]["processing_stats"]
        print("\nConversion complete!")
        print(
            f"Processed {stats['documents_processed']} documents ({stats['documents_skipped']} skipped)"
        )
        print(
            f"Processed {stats['entities_processed']} entities ({stats['entities_skipped']} skipped)"
        )
        print(f"Successfully converted {stats['entities_converted']} entities")
        print(f"All entities flagged for review as false negatives")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
