import time
from utils.llm import ModelLoader
import sys
from typing import List
import torch

class LocalLLMClient:
    def __init__(self, model_type="llama3_8b", device="cuda", 
                 cache_dir="/shared/rsaas/jw3/rare_disease/model_cache", 
                 temperature=0.6):
        """
        Initialize the LLM client.
        
        Args:
            model_type (str): Type of model to use
            device: Device to use, can be a string like "cuda:0" or a list of device strings
            cache_dir (str): Directory to cache models
            temperature (float): Temperature for sampling
        """
        import torch
        
        self.model_loader = ModelLoader(cache_dir=cache_dir)
        self.temperature = temperature
        
        # Handle device parameter which could be a string or a list
        if isinstance(device, list):
            # For multiple GPUs, we need to use device_map='auto' rather than a comma-separated string
            print(f"Using multiple devices: {device}")
            # Just pass 'auto' as the device_map for multiple devices
            self.pipeline = self.model_loader.get_llm_pipeline("auto", model_type)
            
            # For tensor operations, use the first device in the list
            if len(device) > 0:
                self.actual_device = torch.device(device[0])
            else:
                self.actual_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # For single device or "auto"
            self.pipeline = self.model_loader.get_llm_pipeline(device, model_type)
            
            # Determine the actual device for tensor operations
            if hasattr(self.pipeline.model, 'device_map') and self.pipeline.model.device_map == 'auto':
                # With auto device mapping, use the first module's device
                for name, module in self.pipeline.model.named_modules():
                    if hasattr(module, 'device') and module.device is not None:
                        self.actual_device = module.device
                        break
                else:
                    # Fallback if no module has a device attribute
                    self.actual_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                # Use the specified device
                self.actual_device = torch.device(device)

    def query(self, user_input, system_message):
        """Send a query to the LLM."""
        import torch
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]

        full_prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        with torch.no_grad():
            outputs = self.pipeline(
                full_prompt,
                max_new_tokens=1000,
                eos_token_id=terminators,
                do_sample=True,
                temperature=self.temperature,
                top_p=1.0,
            )

        return outputs[0]["generated_text"][len(full_prompt):]
    
    def batched_query(self, user_inputs, system_messages, max_batch_size=None):
        """Process multiple queries in a single GPU batch."""
        import torch
        
        # Ensure inputs and messages have the same length
        if len(user_inputs) != len(system_messages):
            raise ValueError("Length of user_inputs and system_messages must be the same")
        
        # If only one input, just use the regular query method
        if len(user_inputs) == 1:
            return [self.query(user_inputs[0], system_messages[0])]
        
        # Determine batch size and number of batches
        total_samples = len(user_inputs)
        if max_batch_size is None:
            batch_size = total_samples
        else:
            batch_size = min(max_batch_size, total_samples)
        
        num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
        all_results = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            
            batch_user_inputs = user_inputs[start_idx:end_idx]
            batch_system_messages = system_messages[start_idx:end_idx]
            
            # Create formatted prompts for each input in the batch
            batch_prompts = []
            for j in range(len(batch_user_inputs)):
                messages = [
                    {"role": "system", "content": batch_system_messages[j]},
                    {"role": "user", "content": batch_user_inputs[j]}
                ]
                
                full_prompt = self.pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_prompts.append(full_prompt)
            
            # Tokenize all prompts in the batch and pad them
            tokenizer = self.pipeline.tokenizer
            
            # Ensure the padding token is set properly
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Tokenize the batch
            tokenized_inputs = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=2048,  # Limit input length to avoid out-of-memory errors
                return_length=True
            )
            
            # Move inputs to the appropriate device determined during initialization
            for key in tokenized_inputs:
                if isinstance(tokenized_inputs[key], torch.Tensor):
                    tokenized_inputs[key] = tokenized_inputs[key].to(self.actual_device)
            
            # Store the original lengths to extract responses correctly later
            prompt_lengths = tokenized_inputs.pop("length").tolist()
            
            # Setup termination tokens
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.pipeline.model.generate(
                    **tokenized_inputs,
                    max_new_tokens=1000,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=1.0,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode the generated text and extract only the new part (excluding the prompt)
            batch_results = []
            for j, (gen_ids, prompt_len) in enumerate(zip(generated_ids, prompt_lengths)):
                # Extract only the newly generated tokens (excluding the prompt)
                new_tokens = gen_ids[prompt_len:]
                # Decode the tokens
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                batch_results.append(generated_text)
            
            all_results.extend(batch_results)
        
        return all_results
    
    def _calculate_entropy(self, probs):
        """Calculate entropy as a measure of uncertainty"""
        import math
        entropy = 0
        for p in probs:
            if p > 0:  # Avoid log(0)
                entropy -= p * math.log2(p)
        return entropy
    
    def query_with_full_entropy(self, user_input, system_message):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]

        full_prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize the prompt
        input_ids = self.pipeline.tokenizer.encode(full_prompt, return_tensors="pt").to(self.pipeline.model.device)
        prompt_length = input_ids.shape[1]
        
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        with torch.no_grad():
            outputs = self.pipeline.model.generate(
                input_ids,
                max_new_tokens=1000,
                do_sample=True,
                temperature=self.temperature,
                top_p=1.0,
                eos_token_id=terminators,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # Extract generated tokens (excluding prompt)
        generated_sequence = outputs.sequences[0, prompt_length:]
        generated_text = self.pipeline.tokenizer.decode(generated_sequence, skip_special_tokens=True)
        
        # Calculate token probabilities and distribution entropy
        token_probs = []
        distribution_entropies = []
        
        for i, token_scores in enumerate(outputs.scores):
            # Get the next token ID that was actually generated
            next_token_id = generated_sequence[i].item()
            
            # Apply softmax to get full probability distribution
            logits = token_scores[0]
            full_distribution = torch.nn.functional.softmax(logits, dim=-1)
            # CORRECTED: Instead of directly using the token ID as an index,
            # find the probability of the chosen token properly
            selected_token_prob = full_distribution[next_token_id].item()
            
            # Verify our selection matches what was generated
            max_prob_token_id = torch.argmax(logits).item()
            if max_prob_token_id != next_token_id and i < 5:  # Only log first few tokens for debugging
                # This is expected sometimes with temperature > 0, as we sample from the distribution
                print(f"Note: Generated token '{self.pipeline.tokenizer.decode([next_token_id])}' (p={selected_token_prob:.4f}) " 
                    f"differs from highest probability token '{self.pipeline.tokenizer.decode([max_prob_token_id])}' "
                    f"(p={full_distribution[max_prob_token_id].item():.4f})")
            
            token_probs.append(selected_token_prob)
            
            # Calculate entropy of the entire distribution
            distribution_entropy = 0
            # Only consider tokens with non-negligible probability for efficiency
            significant_probs = full_distribution[full_distribution > 1e-5]
            for prob in significant_probs:
                p = prob.item()
                distribution_entropy -= p * torch.log2(torch.tensor(p)).item()
            
            distribution_entropies.append(distribution_entropy)
        
        # Get top alternative tokens for the first few positions
        top_alternatives = []
        for i in range(min(5, len(outputs.scores))):
            token_scores = outputs.scores[i][0]
            selected_id = generated_sequence[i].item()
            
            # Get top 5 tokens
            values, indices = torch.topk(token_scores, 5)
            # Apply softmax to these top values to get probabilities
            probs = torch.nn.functional.softmax(values, dim=-1)
            
            alternatives = []
            for j in range(5):
                token_id = indices[j].item()
                token_text = self.pipeline.tokenizer.decode([token_id])
                # Use the correct probability from the full distribution
                token_prob = full_distribution[token_id].item()
                alternatives.append({
                    "token": token_text,
                    "probability": token_prob,
                    "is_selected": token_id == selected_id
                })
            
            top_alternatives.append(alternatives)
        
        return {
            "generated_text": generated_text,
            "token_probs": token_probs,
            "mean_confidence": sum(token_probs) / len(token_probs) if token_probs else 0,
            "min_confidence": min(token_probs) if token_probs else 0,
            "token_distribution_entropies": distribution_entropies,
            "mean_distribution_entropy": sum(distribution_entropies) / len(distribution_entropies) if distribution_entropies else 0,
            "max_distribution_entropy": max(distribution_entropies) if distribution_entropies else 0,
            "top_alternatives": top_alternatives
        }
    

from typing import Dict, List, Optional, Union, Any
import time
import os
import json
import torch





class VLLMClient:
    """
    LLM client implementation using VLLM for optimized inference.
    
    VLLM offers significant performance improvements over standard HuggingFace
    pipelines, especially for batch inference and when using multiple GPUs.
    """
    
    MODEL_MAPPING = {
        "llama3_70b": "meta-llama/Llama-3-70b-chat-hf",
        "llama3_8b": "meta-llama/Llama-3-8b-chat-hf",
        "llama3_70b_2b": "meta-llama/Llama-3-70b-chat-hf",
        "mistral_24b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama3_70b_r1": "meta-llama/Llama-3-70b-chat-hf",
        "qwen_70b": "Qwen/Qwen2-72B-Instruct",
        "mixtral_70b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }
    
    # Map of models to their parameter sizes in billions
    MODEL_SIZES = {
        "llama3_70b": 70,
        "llama3_8b": 8,
        "llama3_70b_2b": 70,
        "mistral_24b": 24,
        "llama3_70b_r1": 70,
        "qwen_70b": 72,
        "mixtral_70b": 70,
    }
    
    def __init__(
        self,
        model_type: str = "llama3_8b",
        device: Union[str, List[str]] = "cuda",
        cache_dir: Optional[str] = None,
        temperature: float = 0.7,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        quantization: Optional[str] = None
    ):
        """
        Initialize a VLLM-based LLM client.
        
        Args:
            model_type: Type of model to use (mapped to HuggingFace model names)
            device: Device(s) to use - can be string like "cuda:0" or list of devices
            cache_dir: Directory to cache models
            temperature: Temperature for sampling
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            quantization: Quantization method (None, '4bit', or '8bit')
        """
        self.model_type = model_type
        self.device = device
        self.cache_dir = cache_dir
        self.temperature = temperature
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        
        # Auto-select 4-bit quantization for models larger than 8B parameters
        if quantization is None:
            model_size = self.MODEL_SIZES.get(model_type, 0)
            if model_size > 8:
                quantization = "4bit"
                print(f"Auto-selecting 4-bit quantization for {model_type} ({model_size}B parameters)")
        
        self.quantization = quantization
        
        # Will be set when model is loaded
        self.llm = None
        self.tokenizer = None
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the model using VLLM."""
        try:
            from vllm import LLM, SamplingParams
            from vllm.outputs import RequestOutput
        except ImportError:
            raise ImportError(
                "VLLM is not installed. Install it with 'pip install vllm'"
            )
        
        # Get the model name from mapping
        model_name = self.MODEL_MAPPING.get(self.model_type, self.model_type)
        
        # Prepare VLLM-specific kwargs
        vllm_kwargs = {
            "model": model_name,
            "download_dir": self.cache_dir,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
        }
        
        # Handle device configuration
        if isinstance(self.device, list):
            # If device is a list, use the first n devices for tensor parallelism
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [d.split(":")[-1] if ":" in d else d for d in self.device[:self.tensor_parallel_size]]
            )
        elif self.device == "auto":
            # Use all available GPUs
            vllm_kwargs["tensor_parallel_size"] = torch.cuda.device_count()
        elif isinstance(self.device, str) and self.device != "cuda":
            # Specific device
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device.split(":")[-1] if ":" in self.device else "0"
        
        # Set quantization if requested
        if self.quantization:
            if self.quantization == "4bit":
                vllm_kwargs["quantization"] = "awq"  # VLLM supports AWQ for 4-bit
            elif self.quantization == "8bit":
                vllm_kwargs["quantization"] = "int8"  # 8-bit quantization
        
        # Log initialization
        print(f"Initializing VLLM with model: {model_name}")
        print(f"Tensor parallelism: {vllm_kwargs['tensor_parallel_size']}")
        if self.quantization:
            print(f"Quantization: {self.quantization}")
        
        # Initialize the VLLM model
        self.llm = LLM(**vllm_kwargs)
        
        # Initialize sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.95,
            max_tokens=4096
        )
        
        # Store the request_output class for later use
        self.RequestOutput = RequestOutput
    
    def _format_prompt(self, user_input: str, system_message: str) -> str:
        """Format the prompt using the appropriate chat template."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]
        
        # Use the VLLM model's tokenizer to apply the chat template
        return self.llm.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def query(self, user_input: str, system_message: str) -> str:
        """
        Send a query to the LLM using VLLM.
        
        Args:
            user_input: User's input text
            system_message: System message for context
            
        Returns:
            Model's response text
        """
        if self.llm is None:
            raise RuntimeError("Model not initialized. Call _load_model first.")
        
        # Format the prompt using the model's chat template
        prompt = self._format_prompt(user_input, system_message)
        
        # Generate response
        outputs = self.llm.generate(prompt, self.sampling_params)
        
        # Extract and return the generated text
        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text
        return ""
    
    def batch_query(self, queries: List[str], system_message: str) -> List[str]:
        """
        Process multiple queries efficiently in a single batch.
        
        Args:
            queries: List of user queries to process
            system_message: System message to use for all queries
            
        Returns:
            List of model responses
        """
        if self.llm is None:
            raise RuntimeError("Model not initialized. Call _load_model first.")
        
        # Format all prompts
        prompts = [self._format_prompt(query, system_message) for query in queries]
        
        # Generate responses in a batch
        batch_outputs = self.llm.generate(prompts, self.sampling_params)
        
        # Extract and return generated texts
        results = []
        for output in batch_outputs:
            text = output.outputs[0].text if output.outputs else ""
            results.append(text)
        
        return results

    @classmethod
    def from_config(cls, config_file: str) -> 'VLLMClient':
        """Create a VLLMClient instance from a configuration file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} not found")
            
        with open(config_file, "r") as f:
            config = json.load(f)
            
        return cls(**config)


    
# API LLM Client
from typing import Dict, Any, Optional, List
import time
import json
import os
from datetime import datetime
from functools import wraps
from ratelimit import limits, sleep_and_retry
from groq import Groq
from groq.types.chat import ChatCompletion
from groq.types.chat.chat_completion_message import ChatCompletionMessage

# Rate limits for API calls
API_CALLS_LIMIT = 30
API_PERIOD = 60  # 30 calls per minute
API_MAX_TOKENS = 100000

def rate_limited_api(func):
    """Decorator for API rate limiting."""
    @sleep_and_retry
    @limits(calls=API_CALLS_LIMIT, period=API_PERIOD)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class APILLMClient:
    """API-based LLM client implementation using official Groq client."""
    
    # Updated model mapping for Llama 3.3
    MODEL_MAPPING = {
        "llama3-groq-70b": "llama-3.3-70b-versatile",  # Latest Llama 3.3
        "llama3-groq-70b-chat": "llama-3.3-70b-chat",  # Chat-optimized variant
        "llama3-groq-32b": "llama-3.3-32b-versatile",  # 32B parameter version
        "mixtral-8x7b": "mixtral-8x7b-v0.1",          # Mixtral model
    }
    
    def __init__(
        self,
        model_type: str = "llama3-groq-70b",
        device: str = "cpu",  # Kept for interface compatibility
        cache_dir: str = None,  # Kept for interface compatibility
        max_tokens_per_day: int = API_MAX_TOKENS,
        max_queries_per_minute: int = API_CALLS_LIMIT,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the API LLM client using Groq's official client.
        
        Args:
            model_type: Type of model to use (will be mapped to Groq model names)
            device: Device to use (kept for interface compatibility)
            cache_dir: Cache directory (kept for interface compatibility)
            max_tokens_per_day: Maximum tokens allowed per day
            max_queries_per_minute: Maximum queries allowed per minute
            temperature: Temperature for model responses
            api_key: API key for authentication
        """
        self.model_type = model_type
        self.device = device
        self.cache_dir = cache_dir
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.total_tokens_used = 0
        self.temperature = temperature
        
        # Initialize Groq client
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either through constructor or GROQ_API_KEY environment variable")
        
        self.client = Groq(api_key=self.api_key)
        
        # Save configuration
        self._save_config()

    def _save_config(self) -> None:
        """Save current configuration to file."""
        config = {
            "model_type": self.model_type,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "max_tokens_per_day": self.max_tokens_per_day,
            "max_queries_per_minute": self.max_queries_per_minute,
            "temperature": self.temperature,
            "api_key": self.api_key,
        }
        
        config_file = os.path.join(self.cache_dir, "api_config.json") if self.cache_dir else "api_config.json"
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        with open(config_file, "w") as f:
            json.dump(config, f)

    @classmethod
    def from_config(cls, config_file: str) -> 'APILLMClient':
        """Create an APILLMClient instance from a configuration file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} not found")
            
        with open(config_file, "r") as f:
            config = json.load(f)
            
        return cls(**config)

    @rate_limited_api
    def query(self, user_input: str, system_message: str) -> str:
        """
        Send a query to the Groq API using the official client.
        
        Args:
            user_input: User's input text
            system_message: System message for context
            
        Returns:
            Model's response text
        """
        print(f"TOTAL_TOKENS_USED before query: {self.total_tokens_used}")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]
        
        # Get appropriate model name
        model_name = self.MODEL_MAPPING.get(self.model_type, self.model_type)
        
        try:
            # Create chat completion
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=self.temperature,
            )
            
            # Update token usage if available
            if hasattr(chat_completion, 'usage'):
                self.total_tokens_used += chat_completion.usage.total_tokens
            
            # Extract and return the response
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            print(f"Groq API request failed: {str(e)}")
            raise

    @staticmethod
    def initialize_from_input() -> 'APILLMClient':
        """Initialize client by gathering input from user."""
        print("\nInitializing Groq API Client")
        print("-" * 30)
        print("\nAvailable models:")
        for key, value in APILLMClient.MODEL_MAPPING.items():
            print(f"  {key}: {value}")
        print()
        
        # Gather required inputs
        api_key = input("Enter your API key (or set GROQ_API_KEY env var): ").strip()
        if not api_key and not os.getenv("GROQ_API_KEY"):
            raise ValueError("API key is required")
            
        # Gather optional inputs with defaults
        model_type = input("Enter model type (default 'llama3-groq-70b'): ").strip() or "llama3-groq-70b"
        cache_dir = input("Enter cache directory (optional): ").strip() or None
        
        # Use API-specific defaults
        max_tokens = input(f"Enter max tokens per day (default {API_MAX_TOKENS}): ").strip()
        max_tokens = int(max_tokens) if max_tokens else API_MAX_TOKENS
        max_queries = input(f"Enter max queries per minute (default {API_CALLS_LIMIT}): ").strip()
        max_queries = int(max_queries) if max_queries else API_CALLS_LIMIT
        
        temperature = input("Enter temperature (default 0.7, range 0.0-1.0): ").strip()
        try:
            temperature = float(temperature) if temperature else 0.7
            if not 0.0 <= temperature <= 1.0:
                raise ValueError
        except ValueError:
            print("Invalid temperature. Using default (0.7)")
            temperature = 0.7
            
        return APILLMClient(
            api_key=api_key,
            model_type=model_type,
            cache_dir=cache_dir,
            max_tokens_per_day=max_tokens,
            max_queries_per_minute=max_queries,
            temperature=temperature
        )