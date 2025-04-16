from utils.data import read_json_file, print_json_structure
import re
import json

def load_mistral_llm_client():
    """
    Load a Mistral 24B LLM client configured with default cache directories
    and assigned to cuda:0 device.
    
    Returns:
        LocalLLMClient: Initialized LLM client for Mistral 24B
    """
    from utils.llm_client import LocalLLMClient
    
    # Default cache directory from mine_hpo.py
    default_cache_dir = "/shared/rsaas/jw3/rare_disease/model_cache"
    
    # Initialize and return the client with specific configuration
    llm_client = LocalLLMClient(
        model_type="mistral_24b",  # Explicitly request mistral_24b model
        device="cuda:3",           # Assign to first GPU (cuda:0)
        cache_dir=default_cache_dir,
        temperature=0.0001           # Default temperature from mine_hpo.py
    )
    
    return llm_client

def process_page_content(content, llm_client, system_message):
    """Process a single page's content using the LLM client with robust parsing"""
    response = llm_client.query(content, system_message)
    
    # Try to extract JSON from markdown code blocks if present
    json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    json_match = re.search(json_pattern, response)
    
    if json_match:
        try:
            # Parse the JSON from inside the code block
            json_str = json_match.group(1)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If no code block or parsing failed, try parsing the whole response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # If all parsing attempts fail, return raw response
        return {"raw_response": response}

def extract_tables_from_all_pages(file_path, llm_client):
    """Process all pages in the file and convert tables to JSON"""
    # Read the data file
    data = read_json_file(file_path)
    
    # System message to guide the LLM
    system_message = """
    Extract all the corresponding tables for each "test number" into a JSON format.
    For each test:
    1. Include the test number as the key
    2. Include the test name
    3. Structure the reference ranges by age group, with male and female values

    IMPORTANT: Return ONLY the raw JSON object. Do not include markdown code blocks, backticks, or any other formatting.
    Example format:
    {
    "123231": {
        "name": "ACE, CSF",
        "reference_ranges": [
        {"age_group": "0-5 years", "male": "Not Estab.", "female": "Not Estab."},
        {"age_group": "6-17 years", "male": "0.0-2.1", "female": "0.0-2.1"}
        ],
        "units": "U/L"
    }
    }
    """
    
    # Process each page
    all_results = {}
    
    for i, page in enumerate(data):
        print(f"Processing page {i+1}/{len(data)}...")
        content = page["content"]
        
        # Skip pages without tables or test numbers
        if "Test Number:" not in content:
            continue
            
        # Process the page
        page_results = process_page_content(content, llm_client, system_message)
        
        # Merge results
        all_results.update(page_results)
    
    return all_results

# Main execution
if __name__ == "__main__":
    file_path = "data/tools/extracted_pages.json"

    llm_client = load_mistral_llm_client()
    # Process all pages
    results = extract_tables_from_all_pages(file_path, llm_client)
    
    # Save results to a new JSON file
    output_path = "data/tools/processed_lab_tables.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Results saved to {output_path}")
    print(f"Total test numbers extracted: {len(results)}")