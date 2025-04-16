import json
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class CodeMetadata:
    """Store metadata for a code including labels and related codes."""
    label: str
    icd10_codes: List[str] = None
    icd10_labels: List[str] = None

@dataclass
class OntologyMappings:
    """Store all mappings between different ontologies."""
    umls_to_hpo: Dict[str, Set[str]]
    hpo_to_umls: Dict[str, Set[str]]
    umls_to_orpha: Dict[str, Set[str]]
    orpha_to_umls: Dict[str, Set[str]]
    # Metadata storage
    hpo_metadata: Dict[str, CodeMetadata]
    orpha_metadata: Dict[str, CodeMetadata]

class OntologyConverter:
    def __init__(self, hpo_file: str, ordo_file: str):
        """
        Initialize the converter with HPO and ORDO ontology files.
        
        Args:
            hpo_file: Path to the HPO JSON file
            ordo_file: Path to the ORDO JSON file
        """
        self.mappings = self._load_mappings(hpo_file, ordo_file)
    
    def _load_mappings(self, hpo_file: str, ordo_file: str) -> OntologyMappings:
        """
        Load and process both ontology files to create mapping dictionaries.
        
        Args:
            hpo_file: Path to the HPO JSON file
            ordo_file: Path to the ORDO JSON file
            
        Returns:
            OntologyMappings object containing all mappings
            
        Raises:
            FileNotFoundError: If either file doesn't exist
            UnicodeDecodeError: If files contain encoding issues
            json.JSONDecodeError: If files contain invalid JSON
        """
        # Initialize mapping dictionaries
        umls_to_hpo = defaultdict(set)
        hpo_to_umls = defaultdict(set)
        umls_to_orpha = defaultdict(set)
        orpha_to_umls = defaultdict(set)
        # Initialize metadata dictionaries
        hpo_metadata = {}
        orpha_metadata = {}
        
        # Process HPO file
        try:
            with open(hpo_file, 'r', encoding='utf-8') as f:
                hpo_data = json.load(f)
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                f"Error reading {hpo_file}: File contains non-UTF-8 characters. "
                f"Original error: {str(e)}"
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Error parsing {hpo_file}: Invalid JSON format. "
                f"Error at line {e.lineno}, column {e.colno}: {e.msg}",
                e.doc,
                e.pos
            )
            
        for node in hpo_data['graphs'][0]['nodes']:
            hpo_id = node['id'].replace('http://purl.obolibrary.org/obo/', '')
            
            # Store HPO metadata
            hpo_metadata[hpo_id] = CodeMetadata(
                label=node.get('lbl', '')
            )
            
            if 'meta' in node and 'xrefs' in node['meta']:
                for xref in node['meta']['xrefs']:
                    if xref['val'].startswith('UMLS:'):
                        umls_cui = xref['val'].replace('UMLS:', '')
                        umls_to_hpo[umls_cui].add(hpo_id)
                        hpo_to_umls[hpo_id].add(umls_cui)
        
        # Process ORDO file
        try:
            with open(ordo_file, 'r', encoding='utf-8') as f:
                ordo_data = json.load(f)
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                f"Error reading {ordo_file}: File contains non-UTF-8 characters. "
                f"Original error: {str(e)}"
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Error parsing {ordo_file}: Invalid JSON format. "
                f"Error at line {e.lineno}, column {e.colno}: {e.msg}",
                e.doc,
                e.pos
            )
            
        for entry in ordo_data['Sheet1']:
            orpha_id = entry['ORDO ID'].replace('http://www.orpha.net/ORDO/Orphanet_', '')
            
            # Store ORPHA metadata
            icd10_codes = eval(entry['ICD IDs']) if entry['ICD IDs'] else []
            orpha_metadata[orpha_id] = CodeMetadata(
                label=entry['Preferred Label'],
                icd10_codes=icd10_codes,
                icd10_labels=[]  # You would need to add ICD10 labels from another source
            )
            
            umls_ids = eval(entry['UMLS IDs'])  # Safely evaluate string representation of list
            for umls_id in umls_ids:
                if umls_id.startswith('UMLS:'):
                    umls_cui = umls_id.replace('UMLS:', '')
                    umls_to_orpha[umls_cui].add(orpha_id)
                    orpha_to_umls[orpha_id].add(umls_cui)
        
        return OntologyMappings(
            umls_to_hpo=dict(umls_to_hpo),
            hpo_to_umls=dict(hpo_to_umls),
            umls_to_orpha=dict(umls_to_orpha),
            orpha_to_umls=dict(orpha_to_umls),
            hpo_metadata=hpo_metadata,
            orpha_metadata=orpha_metadata
        )
    
    def get_code_metadata(self, code: str) -> Optional[CodeMetadata]:
        """
        Get metadata for a specific code (HPO or ORPHA).
        
        Args:
            code: The code to look up metadata for
            
        Returns:
            CodeMetadata object if found, None otherwise
        """
        if code.startswith('HP_'):
            return self.mappings.hpo_metadata.get(code)
        elif code.isdigit():  # ORPHA code
            return self.mappings.orpha_metadata.get(code)
        return None
    
    def check_and_convert_umls(self, umls_cui: str) -> Dict[str, Any]:
        """
        Check if a UMLS CUI exists in HPO and/or ORPHA mappings and convert if present.
        Also returns associated metadata for validation.
        
        Args:
            umls_cui: The UMLS CUI to check and convert
            
        Returns:
            Dictionary containing:
            - 'in_hpo': Boolean indicating if CUI exists in HPO mappings
            - 'in_orpha': Boolean indicating if CUI exists in ORPHA mappings
            - 'hpo_codes': Set of mapped HPO codes (if any)
            - 'orpha_codes': Set of mapped ORPHA codes (if any)
            - 'hpo_metadata': List of metadata for mapped HPO codes
            - 'orpha_metadata': List of metadata for mapped ORPHA codes
        """
        # Strip 'UMLS:' prefix if present
        umls_cui = umls_cui.replace('UMLS:', '')
        
        result = {
            'in_hpo': False,
            'in_orpha': False,
            'hpo_codes': set(),
            'orpha_codes': set(),
            'hpo_metadata': [],
            'orpha_metadata': []
        }
        
        # Check HPO mappings
        if umls_cui in self.mappings.umls_to_hpo:
            result['in_hpo'] = True
            hpo_codes = self.mappings.umls_to_hpo[umls_cui]
            result['hpo_codes'] = hpo_codes
            # Add metadata for each HPO code
            for hpo_code in hpo_codes:
                if metadata := self.mappings.hpo_metadata.get(hpo_code):
                    result['hpo_metadata'].append({
                        'code': hpo_code,
                        'metadata': metadata
                    })
            
        # Check ORPHA mappings
        if umls_cui in self.mappings.umls_to_orpha:
            result['in_orpha'] = True
            orpha_codes = self.mappings.umls_to_orpha[umls_cui]
            result['orpha_codes'] = orpha_codes
            # Add metadata for each ORPHA code
            for orpha_code in orpha_codes:
                if metadata := self.mappings.orpha_metadata.get(orpha_code):
                    result['orpha_metadata'].append({
                        'code': orpha_code,
                        'metadata': metadata
                    })
            
        return result

    def get_all_mappings(self, code: str) -> Dict[str, Any]:
        """
        Get all available mappings for a given code.
        Automatically detects the code type and returns all related codes with metadata.
        
        Args:
            code: The input code (UMLS, HPO, or ORPHA)
            
        Returns:
            Dictionary containing all available mappings and metadata
        """
        mappings = {
            'codes': {},
            'metadata': {}
        }
        
        # Try as UMLS
        if code.startswith('C'):
            result = self.check_and_convert_umls(code)
            if result['hpo_codes']:
                mappings['codes']['HPO'] = result['hpo_codes']
                mappings['metadata']['HPO'] = result['hpo_metadata']
            if result['orpha_codes']:
                mappings['codes']['ORPHA'] = result['orpha_codes']
                mappings['metadata']['ORPHA'] = result['orpha_metadata']
                
        # Try as HPO
        elif code.startswith('HP_'):
            umls_codes = self.hpo_to_umls(code)
            if umls_codes:
                mappings['codes']['UMLS'] = umls_codes
                # Get HPO metadata
                if metadata := self.get_code_metadata(code):
                    mappings['metadata']['HPO'] = [{
                        'code': code,
                        'metadata': metadata
                    }]
                # Get related ORPHA codes through UMLS
                orpha_metadata = []
                orpha_codes = set()
                for umls_code in umls_codes:
                    orpha_result = self.check_and_convert_umls(umls_code)
                    orpha_codes.update(orpha_result['orpha_codes'])
                    orpha_metadata.extend(orpha_result['orpha_metadata'])
                if orpha_codes:
                    mappings['codes']['ORPHA'] = orpha_codes
                    mappings['metadata']['ORPHA'] = orpha_metadata
                    
        # Try as ORPHA
        elif code.isdigit():
            umls_codes = self.orpha_to_umls(code)
            if umls_codes:
                mappings['codes']['UMLS'] = umls_codes
                # Get ORPHA metadata
                if metadata := self.get_code_metadata(code):
                    mappings['metadata']['ORPHA'] = [{
                        'code': code,
                        'metadata': metadata
                    }]
                # Get related HPO codes through UMLS
                hpo_metadata = []
                hpo_codes = set()
                for umls_code in umls_codes:
                    hpo_result = self.check_and_convert_umls(umls_code)
                    hpo_codes.update(hpo_result['hpo_codes'])
                    hpo_metadata.extend(hpo_result['hpo_metadata'])
                if hpo_codes:
                    mappings['codes']['HPO'] = hpo_codes
                    mappings['metadata']['HPO'] = hpo_metadata
                    
        return mappings

    def analyze_unmapped_codes(self) -> Dict[str, Any]:
        """
        Analyze HPO and ORPHA codes that don't have UMLS mappings.
        
        Returns:
            Dictionary containing:
            - unmapped_hpo: List of HPO codes without UMLS mappings
            - unmapped_orpha: List of ORPHA codes without UMLS mappings
            Each entry includes the code and its metadata
        """
        unmapped_hpo = []
        unmapped_orpha = []
        
        # Check HPO codes
        for hpo_id, metadata in self.mappings.hpo_metadata.items():
            if hpo_id not in self.mappings.hpo_to_umls:
                unmapped_hpo.append({
                    'code': hpo_id,
                    'label': metadata.label
                })
        
        # Check ORPHA codes
        for orpha_id, metadata in self.mappings.orpha_metadata.items():
            if orpha_id not in self.mappings.orpha_to_umls:
                unmapped_orpha.append({
                    'code': orpha_id,
                    'label': metadata.label,
                    'icd10_codes': metadata.icd10_codes
                })
        
        return {
            'unmapped_hpo': unmapped_hpo,
            'unmapped_orpha': unmapped_orpha,
            'stats': {
                'total_hpo': len(self.mappings.hpo_metadata),
                'unmapped_hpo': len(unmapped_hpo),
                'hpo_unmapped_percentage': round(len(unmapped_hpo) / len(self.mappings.hpo_metadata) * 100, 2),
                'total_orpha': len(self.mappings.orpha_metadata),
                'unmapped_orpha': len(unmapped_orpha),
                'orpha_unmapped_percentage': round(len(unmapped_orpha) / len(self.mappings.orpha_metadata) * 100, 2)
            }
        }

# Example usage
if __name__ == "__main__":
    converter = OntologyConverter("export/ontology/hpo/hp.json", "export/ontology/ordo/ORDO2_UMLS_ICD.json")
    
    # Analyze unmapped codes
    print("\nAnalyzing unmapped codes:")
    print("=" * 50)
    
    analysis = converter.analyze_unmapped_codes()
    
    # Print statistics
    stats = analysis['stats']
    print("\nStatistics:")
    print(f"HPO Terms: {stats['total_hpo']}")
    print(f"Unmapped HPO Terms: {stats['unmapped_hpo']} ({stats['hpo_unmapped_percentage']}%)")
    print(f"ORPHA Terms: {stats['total_orpha']}")
    print(f"Unmapped ORPHA Terms: {stats['unmapped_orpha']} ({stats['orpha_unmapped_percentage']}%)")
    
    # Print sample of unmapped HPO terms
    print("\nSample of unmapped HPO terms:")
    for entry in analysis['unmapped_hpo'][:5]:
        print(f"- {entry['code']}: {entry['label']}")
    
    # Print sample of unmapped ORPHA terms
    print("\nSample of unmapped ORPHA terms:")
    for entry in analysis['unmapped_orpha'][:5]:
        print(f"- {entry['code']}: {entry['label']}")
        if entry['icd10_codes']:
            print(f"  ICD-10: {entry['icd10_codes']}")

    # Test cases
    test_cases = [
        "C0037769",  # Should be in both HPO and ORPHA
        "C0000000",  # Should be in neither
        "C0444868",  # Should be in HPO only
    ]
    
    print("\nTesting UMLS code presence, conversion, and metadata:")
    print("=" * 50)
    for umls_cui in test_cases:
        print(f"\nChecking UMLS CUI: {umls_cui}")
        result = converter.check_and_convert_umls(umls_cui)
        
        # Display HPO results
        print(f"Present in HPO: {result['in_hpo']}")
        if result['in_hpo']:
            print("HPO mappings:")
            for metadata in result['hpo_metadata']:
                print(f"  - {metadata['code']}: {metadata['metadata'].label}")
        
        # Display ORPHA results
        print(f"Present in ORPHA: {result['in_orpha']}")
        if result['in_orpha']:
            print("ORPHA mappings:")
            for metadata in result['orpha_metadata']:
                print(f"  - {metadata['code']}: {metadata['metadata'].label}")
                if metadata['metadata'].icd10_codes:
                    print(f"    ICD-10 codes: {metadata['metadata'].icd10_codes}")