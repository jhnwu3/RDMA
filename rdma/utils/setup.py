import argparse
import json
import os
import torch
import traceback
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
from rdma.utils.llm_client import LocalLLMClient, APILLMClient
def timestamp_print(message: str) -> None:
    """Print message with timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def setup_device(args: argparse.Namespace) -> Dict[str, str]:
    """Configure devices for LLM and embeddings based on command line arguments."""
    # Initialize result dictionary
    devices = {}
    
    # Configure LLM device
    if args.cpu:
        devices["llm"] = "cpu"
    elif args.condor:
        if torch.cuda.is_available():
            timestamp_print("Using generic CUDA device for LLM in condor/job scheduler environment")
            devices["llm"] = "cuda"
        else:
            timestamp_print("Warning: CUDA requested but not available. Falling back to CPU for LLM.")
            devices["llm"] = "cpu"
    elif args.gpu_id is not None:
        if torch.cuda.is_available():
            if args.gpu_id < torch.cuda.device_count():
                devices["llm"] = f"cuda:{args.gpu_id}"
            else:
                timestamp_print(f"Warning: GPU {args.gpu_id} requested but only {torch.cuda.device_count()} GPUs available. Using GPU 0 for LLM.")
                devices["llm"] = "cuda:0"
        else:
            timestamp_print(f"Warning: GPU {args.gpu_id} requested but no CUDA available. Falling back to CPU for LLM.")
            devices["llm"] = "cpu"
    else:
        devices["llm"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure embeddings device - handle both attribute names
    # First check if retriever-specific arguments exist in the namespace
    has_retriever_args = hasattr(args, 'retriever_cpu') or hasattr(args, 'retriever_gpu_id')
    
    if has_retriever_args:
        # Using explicit retriever device settings if available
        if hasattr(args, 'retriever_cpu') and args.retriever_cpu:
            devices["embeddings"] = "cpu"
        elif hasattr(args, 'retriever_gpu_id') and args.retriever_gpu_id is not None:
            if torch.cuda.is_available():
                if args.retriever_gpu_id < torch.cuda.device_count():
                    devices["embeddings"] = f"cuda:{args.retriever_gpu_id}"
                else:
                    timestamp_print(f"Warning: GPU {args.retriever_gpu_id} requested for embeddings but only {torch.cuda.device_count()} GPUs available. Using GPU 0.")
                    devices["embeddings"] = "cuda:0"
            else:
                timestamp_print(f"Warning: GPU requested for embeddings but no CUDA available. Falling back to CPU.")
                devices["embeddings"] = "cpu"
        else:
            # Default to using the same device as LLM if retriever args exist but neither is set
            devices["embeddings"] = devices["llm"]
    else:
        # No retriever-specific args, so use the same device as LLM
        devices["embeddings"] = devices["llm"]
        timestamp_print("No embeddings-specific device arguments found. Using same device as LLM for embeddings.")
    
    # For backward compatibility - provide "retriever" key pointing to the same device
    devices["retriever"] = devices["embeddings"]
    
    return devices


# In step3_match_rd.py, change this function:

def initialize_llm_client(args: argparse.Namespace, device_info: Dict[str, str]):
    """Initialize appropriate LLM client based on arguments."""
    # Extract the LLM device from the device_info dictionary
    device = device_info['llm']
    
    if args.llm_type == "api":
        if args.api_config:
            return APILLMClient.from_config(args.api_config)
        else:
            return APILLMClient.initialize_from_input()
    else:  # local
        return LocalLLMClient(
            model_type=args.model_type,
            device=device,  # Pass the string, not the dictionary
            cache_dir=args.cache_dir,
            temperature=args.temperature
        )