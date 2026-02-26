import json
import re
import sys
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing extra spaces, punctuation, and converting to lowercase.
    """
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r"\s+", " ", text.strip().lower())
    # Remove common punctuation but keep letters, numbers, and basic punctuation
    text = re.sub(r"[^\w\s\-\(\)/]", "", text)
    return text


def is_similar(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """
    Check if two texts are similar using sequence matching.
    """
    normalized1 = normalize_text(text1)
    normalized2 = normalize_text(text2)

    # Direct match
    if normalized1 == normalized2:
        return True

    # Check if one is contained in the other
    if normalized1 in normalized2 or normalized2 in normalized1:
        return True

    # Sequence similarity
    similarity = SequenceMatcher(None, normalized1, normalized2).ratio()
    return similarity >= threshold


def contains_phenotype(predicted_disease: str, phenotypes: List[str]) -> bool:
    """
    Check if a predicted disease entry contains or is similar to any phenotype.
    """
    for phenotype in phenotypes:
        if is_similar(predicted_disease, phenotype):
            return True
    return False


def is_analysis_text(text: str) -> bool:
    """
    Check if the text is analytical/reasoning text rather than a disease name.
    """
    analysis_indicators = [
        "analysis",
        "we need to",
        "think of",
        "consider",
        "could be",
        "also consider",
        "but",
        "however",
        "given",
        "features",
        "symptoms",
        "phenotypes",
        "rare diseases that",
        "top 10",
        "most likely",
        "paraneoplastic",
        "autoimmune disorders like",
        "lymphoproliferative disorders",
        "metabolic disorders",
        "bone marrow failure",
    ]

    normalized = normalize_text(text)

    # Check for analysis indicators
    for indicator in analysis_indicators:
        if indicator in normalized:
            return True

    # Check if it's very long (likely analysis)
    if len(text) > 200:
        return True

    return False


def extract_disease_names(predicted_diseases: List[str]) -> List[str]:
    """
    Extract actual disease names from the predicted diseases list,
    filtering out analysis text and phenotypes.
    """
    disease_names = []

    for item in predicted_diseases:
        # Skip analysis text
        if is_analysis_text(item):
            continue

        # Skip very short items (likely fragments)
        if len(item.strip()) < 3:
            continue

        # Skip items that are clearly not disease names
        skip_patterns = [
            r"^[a-z]+$",  # Single lowercase words
            r"^\d+$",  # Just numbers
            r"^etc\.?$",  # "etc"
            r"^and$",  # "and"
            r"^or$",  # "or"
            r"^but$",  # "but"
            r"^the$",  # "the"
            r"^a$",  # "a"
            r"^an$",  # "an"
            r"^not$",  # "not"
            r"^is$",  # "is"
            r"^are$",  # "are"
        ]

        if any(
            re.match(pattern, item.strip(), re.IGNORECASE) for pattern in skip_patterns
        ):
            continue

        disease_names.append(item.strip())

    return disease_names


def filter_predicted_diseases(
    phenotypes: List[str],
    predicted_diseases: List[str],
    similarity_threshold: float = 0.8,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Filter predicted diseases by removing analysis text and phenotype duplicates.
    """
    if debug:
        print(f"    Original predicted diseases: {len(predicted_diseases)}")

    # First, extract actual disease names (filter out analysis text)
    disease_names = extract_disease_names(predicted_diseases)

    if debug:
        print(f"    After analysis filtering: {len(disease_names)}")

    # Then filter out phenotypes
    filtered_diseases = []
    removed_items = []

    for disease in disease_names:
        if not contains_phenotype(disease, phenotypes):
            filtered_diseases.append(disease)
        else:
            removed_items.append(disease)

    if debug:
        print(f"    After phenotype filtering: {len(filtered_diseases)}")
        print(f"    Removed phenotype matches: {len(removed_items)}")

    return {
        "filtered_diseases": filtered_diseases,
        "removed_items": removed_items,
        "original_count": len(predicted_diseases),
        "filtered_count": len(filtered_diseases),
        "removed_count": len(removed_items),
    }


def parse_medical_records(
    text: str,
    filter_predictions: bool = True,
    similarity_threshold: float = 0.8,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Parse medical records text into structured JSON format with optional filtering.

    Args:
        text: Raw text containing patient records
        filter_predictions: Whether to filter predicted diseases
        similarity_threshold: Similarity threshold for phenotype matching
        debug: Enable debug output

    Returns:
        List of dictionaries containing parsed patient data
    """

    # Split text into individual patient records
    patient_sections = text.split("Patient ID: ")[1:]  # Skip first empty split

    if debug:
        print(f"Found {len(patient_sections)} patient sections")

    patients = []
    total_filtering_stats = {
        "original_predictions": 0,
        "filtered_predictions": 0,
        "removed_items": 0,
    }

    for idx, section in enumerate(patient_sections):
        if debug:
            print(f"Processing patient {idx + 1}...")

        patient_data = {}

        # Extract Patient ID
        lines = section.strip().split("\n")
        patient_id_line = lines[0]
        patient_data["patient_id"] = patient_id_line.strip()

        if debug:
            print(f"  Patient ID: {patient_data['patient_id']}")

        # Find the section boundaries
        phenotypes_start = None
        predicted_start = None
        observed_start = None
        hits_start = None

        for i, line in enumerate(lines):
            if line.startswith("Phenotypes: "):
                phenotypes_start = i
            elif line.startswith("Predicted diseases: "):
                predicted_start = i
            elif line.startswith("Observed diseases: "):
                observed_start = i
            elif "Patient hits@K:" in line:
                hits_start = i

        # Extract phenotypes
        if phenotypes_start is not None and predicted_start is not None:
            phenotypes_text = " ".join(lines[phenotypes_start:predicted_start])
            phenotypes_text = phenotypes_text.replace("Phenotypes: ", "").strip()
            # Split by comma and clean up each phenotype
            phenotypes = [p.strip() for p in phenotypes_text.split(", ") if p.strip()]
            patient_data["phenotypes"] = phenotypes

            if debug:
                print(f"  Found {len(phenotypes)} phenotypes")
        else:
            phenotypes = []
            patient_data["phenotypes"] = phenotypes

        # Extract predicted diseases
        raw_predicted_diseases = []
        if predicted_start is not None and observed_start is not None:
            predicted_text = " ".join(lines[predicted_start:observed_start])
            predicted_text = predicted_text.replace("Predicted diseases: ", "").strip()

            # The predicted diseases are in a list format, extract them
            predicted_match = re.search(r"\[(.*?)\]", predicted_text, re.DOTALL)
            if predicted_match:
                predicted_content = predicted_match.group(1)
                # Split by quotes and filter
                items = re.findall(r"'([^']*)'", predicted_content)
                raw_predicted_diseases = [
                    item.strip() for item in items if item.strip()
                ]

                if debug:
                    print(
                        f"  Found {len(raw_predicted_diseases)} raw predicted diseases"
                    )
            else:
                if debug:
                    print("  No predicted diseases found")

        # Apply filtering if requested
        if filter_predictions and phenotypes and raw_predicted_diseases:
            filtering_result = filter_predicted_diseases(
                phenotypes, raw_predicted_diseases, similarity_threshold, debug
            )

            patient_data["predicted_diseases"] = filtering_result["filtered_diseases"]
            patient_data["filtering_stats"] = {
                "original_count": filtering_result["original_count"],
                "filtered_count": filtering_result["filtered_count"],
                "removed_count": filtering_result["removed_count"],
                "removed_items": filtering_result["removed_items"],
            }

            # Update total stats
            total_filtering_stats["original_predictions"] += filtering_result[
                "original_count"
            ]
            total_filtering_stats["filtered_predictions"] += filtering_result[
                "filtered_count"
            ]
            total_filtering_stats["removed_items"] += filtering_result["removed_count"]
        else:
            patient_data["predicted_diseases"] = raw_predicted_diseases
            if filter_predictions:
                patient_data["filtering_stats"] = {
                    "original_count": len(raw_predicted_diseases),
                    "filtered_count": len(raw_predicted_diseases),
                    "removed_count": 0,
                    "removed_items": [],
                }

        # Extract observed diseases
        if observed_start is not None and hits_start is not None:
            observed_text = " ".join(lines[observed_start:hits_start])
            observed_text = observed_text.replace("Observed diseases: ", "").strip()

            # Extract from list format
            observed_match = re.search(r"\[(.*?)\]", observed_text)
            if observed_match:
                observed_content = observed_match.group(1)
                observed_diseases = []
                items = re.findall(r"'([^']*)'", observed_content)
                observed_diseases = [item.strip() for item in items if item.strip()]
                patient_data["observed_diseases"] = observed_diseases

                if debug:
                    print(f"  Found {len(observed_diseases)} observed diseases")
            else:
                patient_data["observed_diseases"] = []
                if debug:
                    print("  No observed diseases found")
        else:
            patient_data["observed_diseases"] = []

        # Extract hits information
        if hits_start is not None:
            hits_line = lines[hits_start]
            hits_match = re.search(r"Patient hits@K: ({.*?})", hits_line)
            if hits_match:
                try:
                    hits_data = eval(
                        hits_match.group(1)
                    )  # Convert string dict to actual dict
                    patient_data["hits_at_k"] = hits_data

                    if debug:
                        print(f"  Hits@K: {hits_data}")
                except:
                    patient_data["hits_at_k"] = {}
                    if debug:
                        print("  Failed to parse hits@K data")

        patients.append(patient_data)

    # Print filtering summary if filtering was applied
    if filter_predictions and total_filtering_stats["original_predictions"] > 0:
        reduction_pct = (
            (
                total_filtering_stats["original_predictions"]
                - total_filtering_stats["filtered_predictions"]
            )
            / total_filtering_stats["original_predictions"]
            * 100
        )

        print(f"\nFiltering Summary:")
        print(
            f"  Original predicted diseases: {total_filtering_stats['original_predictions']}"
        )
        print(
            f"  Filtered predicted diseases: {total_filtering_stats['filtered_predictions']}"
        )
        print(f"  Items removed: {total_filtering_stats['removed_items']}")
        print(f"  Reduction: {reduction_pct:.1f}%")

    return patients


def analyze_hit_statistics(
    input_file: str, skip_lines: int = 48
) -> Optional[Dict[str, Any]]:
    """
    Analyze hit statistics from the medical records file.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Skip the specified number of lines
        content = "".join(lines[skip_lines:])

        # Extract all hits@K data
        hits_pattern = r"Patient hits@K: ({[^}]+})"
        hits_matches = re.findall(hits_pattern, content)

        if not hits_matches:
            print("No hits@K data found")
            return None

        # Parse and aggregate statistics
        total_patients = len(hits_matches)
        k_values = [1, 5, 10]
        stats = {
            "total_patients": total_patients,
            "hits_at_k": {k: {"hits": 0, "percentage": 0.0} for k in k_values},
            "summary": {},
        }

        for hits_str in hits_matches:
            try:
                hits_data = eval(hits_str)
                for k in k_values:
                    if k in hits_data and hits_data[k]:
                        stats["hits_at_k"][k]["hits"] += 1
            except:
                continue

        # Calculate percentages
        for k in k_values:
            hits = stats["hits_at_k"][k]["hits"]
            stats["hits_at_k"][k]["percentage"] = (hits / total_patients) * 100

        # Create summary
        stats["summary"] = {
            f"Hits@{k}": f"{stats['hits_at_k'][k]['hits']}/{total_patients} ({stats['hits_at_k'][k]['percentage']:.1f}%)"
            for k in k_values
        }

        print(f"Hit Statistics for {total_patients} patients:")
        for k in k_values:
            hits = stats["hits_at_k"][k]["hits"]
            pct = stats["hits_at_k"][k]["percentage"]
            print(f"  Hits@{k}: {hits}/{total_patients} ({pct:.1f}%)")

        return stats

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def extract_full_json(
    input_file: str,
    skip_lines: int = 48,
    debug: bool = False,
    filter_predictions: bool = True,
    similarity_threshold: float = 0.8,
) -> Optional[List[Dict[str, Any]]]:
    """
    Extract full patient data as JSON from the medical records file with optional filtering.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Skip the specified number of lines
        content = "".join(lines[skip_lines:])

        if debug:
            print(
                f"Skipped {skip_lines} lines, processing {len(lines) - skip_lines} lines"
            )
            if filter_predictions:
                print(
                    f"Filtering enabled with similarity threshold: {similarity_threshold}"
                )

        # Parse the records
        patients = parse_medical_records(
            content, filter_predictions, similarity_threshold, debug
        )

        return patients

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  python enhanced_parser.py <input_file.out> [mode] [skip_lines] [options]"
        )
        print("")
        print("Modes:")
        print("  stats  - Show hit statistics only (default)")
        print("  json   - Extract full JSON data")
        print("")
        print("Options:")
        print("  debug             - Enable debug output")
        print("  --no-filter       - Disable predicted disease filtering")
        print("  --threshold=X.X   - Set similarity threshold (default: 0.8)")
        print("")
        print("Examples:")
        print("  python enhanced_parser.py results.out")
        print("  python enhanced_parser.py results.out json")
        print("  python enhanced_parser.py results.out json 48")
        print("  python enhanced_parser.py results.out json 48 debug")
        print("  python enhanced_parser.py results.out json 48 --no-filter")
        print("  python enhanced_parser.py results.out json 48 --threshold=0.9")
        sys.exit(1)

    input_file = sys.argv[1]
    mode = "stats"
    skip_lines = 48
    debug = False
    filter_predictions = True
    similarity_threshold = 0.8

    # Parse arguments
    for i in range(2, len(sys.argv)):
        arg = sys.argv[i]

        if arg in ["stats", "json"]:
            mode = arg
        elif arg.isdigit():
            skip_lines = int(arg)
        elif arg.lower() == "debug":
            debug = True
        elif arg == "--no-filter":
            filter_predictions = False
        elif arg.startswith("--threshold="):
            try:
                similarity_threshold = float(arg.split("=")[1])
                if not 0.0 <= similarity_threshold <= 1.0:
                    print("Error: Threshold must be between 0.0 and 1.0")
                    sys.exit(1)
            except ValueError:
                print("Error: Invalid threshold value")
                sys.exit(1)

    if mode == "stats":
        print(f"Analyzing hit statistics from: {input_file}")
        print(f"Skipping first {skip_lines} lines\n")
        stats = analyze_hit_statistics(input_file, skip_lines)

        if stats:
            with open("hit_statistics.json", "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\nStatistics saved to: hit_statistics.json")

    elif mode == "json":
        print(f"Extracting full JSON from: {input_file}")
        print(f"Skipping first {skip_lines} lines")
        if filter_predictions:
            print(f"Filtering enabled (threshold: {similarity_threshold})")
        else:
            print("Filtering disabled")
        if debug:
            print("Debug mode enabled")

        result = extract_full_json(
            input_file, skip_lines, debug, filter_predictions, similarity_threshold
        )
        if result:
            # Generate output filename based on filtering
            base_name = "extracted_patients"
            if filter_predictions:
                output_file = f"{base_name}_filtered.json"
            else:
                output_file = f"{base_name}.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            total_patients = result["summary_metrics"]["total_patients"]
            print(f"Successfully extracted {total_patients} patients to {output_file}")

            # Show summary metrics
            metrics = result["summary_metrics"]
            print(f"\nSummary Metrics:")
            print(f"  Hit Rate: {metrics['hit_rate']:.3f}")
            print(f"  Hit@1 Rate: {metrics['hit_at_1_rate']:.3f}")
            print(f"  Hit@5 Rate: {metrics['hit_at_5_rate']:.3f}")
            print(f"  Hit@10 Rate: {metrics['hit_at_10_rate']:.3f}")
            print(f"  Patient Hit Rate: {metrics['patient_hit_rate']:.3f}")
            print(f"  Parsing Success Rate: {metrics['parsing_success_rate']:.3f}")
            print(f"  Total Diseases: {metrics['total_diseases']}")
            print(f"  Total Patients: {metrics['total_patients']}")

            # Show sample patient
            patient_results = result["patient_results"]
            if patient_results:
                first_patient_id = list(patient_results.keys())[0]
                sample = patient_results[first_patient_id]
                print(f"\nSample patient ({first_patient_id}):")
                print(f"  Phenotypes: {len(sample['phenotypes'].split(', '))} items")
                print(
                    f"  Predicted diseases: {len(sample['predicted_diseases'])} items"
                )
                print(f"  Observed diseases: {len(sample['observed_diseases'])} items")
                print(f"  Hits: {sample['hits']}")

                # Show filtering stats if available
                if "filtering_stats" in sample:
                    stats = sample["filtering_stats"]
                    print(
                        f"  Filtering: {stats['original_count']} → {stats['filtered_count']} ({stats['removed_count']} removed)"
                    )

                # Show first few items of each
                phenotypes_list = sample["phenotypes"].split(", ")
                if phenotypes_list and phenotypes_list[0]:
                    print(f"  First phenotype: {phenotypes_list[0][:100]}...")
                if sample["predicted_diseases"]:
                    print(
                        f"  First predicted disease: {sample['predicted_diseases'][0][:100]}..."
                    )
                if sample["observed_diseases"]:
                    print(f"  First observed disease: {sample['observed_diseases'][0]}")
        else:
            print("No patient data extracted.")

    else:
        print(f"Unknown mode: {mode}. Use 'stats' or 'json'")


if __name__ == "__main__":
    main()
