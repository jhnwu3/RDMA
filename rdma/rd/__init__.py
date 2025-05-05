"""
RDMA: Rare Disease Matching Agent

A Python package for extracting, verifying, and matching rare disease entities 
from clinical text using large language models and embedding-based retrieval.

This package provides a modular pipeline with four main components:
- RDMAExtractor: Extracts rare disease mentions from clinical text
- RDMAVerifier: Verifies if extracted mentions are actual rare diseases
- RDMAMatcher: Matches verified rare diseases to ORPHA codes
- RDMASupervisor: Supervises and refines the results

Author: Claude
"""

__version__ = "0.1.0"

# Import main classes
from rd.extractor import RDMAExtractor
from rd.verifier import RDMAVerifier
from rd.matcher import RDMAMatcher
from rd.supervisor import RDMASupervisor

# Import utility classes
from rdma.utils.llm_client import (
    LocalLLMClient, 
    APILLMClient,
)

# Version information
__all__ = [
    "RDMAExtractor",
    "RDMAVerifier",
    "RDMAMatcher",
    "RDMASupervisor",
    "LocalLLMClient",
    "APILLMClient",
]