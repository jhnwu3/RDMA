import transformers
import torch
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer
from pathlib import Path
import os
from utils.api_keys import LLAMA33_ACCESS_TOKEN, LLAMA31_ACCESS_TOKEN

import os
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig


class ModelLoader:
    def __init__(self, cache_dir="/shared/rsaas/jw3/rare_disease/model_cache"):
        self.cache_dir = Path(os.path.abspath(cache_dir))
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Initialized ModelLoader with cache directory: {self.cache_dir}")

    def get_cache_path(self, model_id, quantization_type):
        model_name = model_id.split('/')[-1]
        
        cache_path = self.cache_dir / f"{model_name}_{quantization_type}"
        if "gemma" in model_id:
            cache_path = self.cache_dir / f"{model_name}"
        
        print(f"Generated cache path: {cache_path}")
        return cache_path

    def is_cached(self, cache_path):
        if not cache_path.exists():
            print(f"Cache directory {cache_path} does not exist")
            return False
            
        safetensors = list(cache_path.glob("*.safetensors"))
        if not safetensors:
            print(f"No safetensors found in {cache_path}")
            return False
            
        config_file = cache_path / "config.json"
        if not config_file.exists():
            print(f"No config.json found in {cache_path}")
            return False
            
        print(f"Valid cache found at {cache_path}")
        return True

    def _setup_device_map(self, devices):
        """
        Set up device mapping based on input devices specification.
        
        Args:
            devices: Can be:
                - 'auto' for automatic mapping
                - a single device string (e.g., 'cuda:0')
                - a list of device strings (e.g., ['cuda:0', 'cuda:1'])
                
        Returns:
            dict or str: Device mapping configuration
        """
        if devices == 'auto':
            return 'auto'
        elif isinstance(devices, str):
            if devices == 'cpu':
                return {'': devices}
            elif 'cuda' in devices:
                # For single GPU, use simple mapping
                return {'': devices}
            else:
                raise ValueError(f"Unsupported device specification: {devices}")
        elif isinstance(devices, list):
            if len(devices) == 1:
                return {'': devices[0]}
            else:
                # For multiple GPUs, let HuggingFace handle the distribution
                return 'auto'
        else:
            raise ValueError(f"Unsupported device specification type: {type(devices)}")

    def get_llm_pipeline(self, devices, model: str):
        """
        Load LLM with support for flexible device configurations
        
        Args:
            devices: Can be:
                - 'auto' for automatic mapping
                - a single device string (e.g., 'cuda:0')
                - a list of device strings (e.g., ['cuda:0', 'cuda:1'])
            model: Model identifier string
        """
        print("Loading LLM!")
        print(f"Device configuration: {devices}")
        
        try:
            device_map = self._setup_device_map(devices)
            print(f"Using device map: {device_map}")
            
            if model == "llama3_70b_full":
                print("Loading full 70B model without quantization")
                return self.load_full_70b_model(device_map)
            elif "70b" in model.lower() or "24b" in model.lower():
                print("Loading 70B model with quantization:", model)
                return self.load_70b_model(device_map, model)
            elif "8b" in model.lower() or "phi4" in model.lower():  # Modified this line to include phi4
                print("Loading 8B model or Phi-4 model:", model)
                return self.load_8b_model(device_map, model) 
            elif model == "gemma3_27b_gptq":
                print("Loading pre-quantized Gemma 3 27B model with GPTQ")
                return self.load_gemma_27b_model(device_map, "")
            elif model == "gemma3_27b_awq":
                print("Loading pre-quantized Gemma 3 27B model with AWQ")
                return self.load_gemma_27b_model(device_map, "awq")
            else:
                raise ValueError(f"Unsupported model type: {model}")
        except Exception as e:
            print(f"Error setting up device mapping: {e}")
            raise

    # Add to utils.llm.py (assuming this file has a ModelLoader class)
    def load_nuextract_model(self, model_name="numind/NuExtract", device="cuda:0", cache_dir=None):
        """
        Load the NuExtract model for structured extraction.
        
        Args:
            model_name: Name of the NuExtract model
            device: Device to load the model on
            cache_dir: Directory to cache the model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"Loading NuExtract model: {model_name} on {device}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=False,
                cache_dir=cache_dir
            )
            
            # Load model with appropriate precision
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=False,
                cache_dir=cache_dir
            )
            
            # Move model to specified device
            device_obj = torch.device(device)
            model.to(device_obj)
            model.eval()
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading NuExtract model: {e}")
            raise

    def load_full_70b_model(self, device_map):
        """Load the 70B model without quantization."""
        model_id = "meta-llama/Llama-3.3-70B-Instruct"
        cache_path = self.get_cache_path(model_id, "full")
        
        if self.is_cached(cache_path):
            print(f"Loading cached full model from {cache_path}")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(cache_path),
                device_map=device_map,
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(str(cache_path))
        else:
            print(f"Model not cached. Loading and caching full model to {cache_path}")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                cache_dir=str(self.cache_dir),
                token=LLAMA33_ACCESS_TOKEN,
                torch_dtype=torch.bfloat16
            )
            model.save_pretrained(str(cache_path))
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                token=LLAMA33_ACCESS_TOKEN,
                cache_dir=str(self.cache_dir)
            )
            tokenizer.save_pretrained(str(cache_path))
        
        self._setup_chat_template(tokenizer)
        
        return self._create_pipeline(model, tokenizer)

    def load_gemma_27b_model(self, device_map, model_type="gptq"):
        """Load pre-quantized Gemma 3 27B model."""
        if model_type == "gptq":
            # model_id = "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g"
            model_id = "unsloth/gemma-3-27b-pt-bnb-4bit"
            model_type = "" # no longer using the gptq
            # model_id = "unsloth/gemma-2-27b-bnb-4bit"
        elif model_type == "awq":
            model_id = "TheBloke/gemma-3-27b-it-AWQ"
            # model_id = "unsloth/gemma-2-27b-bnb-4bit"
        else:
            model_id = "unsloth/gemma-3-27b-pt-bnb-4bit"
            # raise ValueError(f"Unsupported quantization type: {model_type}")
            
        cache_path = self.get_cache_path(model_id, model_type)
        
        if self.is_cached(cache_path):
            print(f"Loading cached pre-quantized model from {cache_path}")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(cache_path),
                device_map=device_map,
                local_files_only=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(str(cache_path))
        else:
            print(f"Model not cached. Loading and caching to {cache_path}")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                cache_dir=str(self.cache_dir),
            )
            model.save_pretrained(str(cache_path))
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                cache_dir=str(self.cache_dir)
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained(str(cache_path))

        self._setup_chat_template(tokenizer)
        
        return self._create_pipeline(model, tokenizer)

    def load_70b_model(self, device_map, model):
        """Load 70B model with quantization."""
        model_id = "aaditya/OpenBioLLM-Llama3-70B"
        if model == "llama3_70b":
            model_id = "meta-llama/Llama-3.3-70B-Instruct"
        elif model == "llama3_70b_groq":
            model_id = "Groq/Llama-3-Groq-70B-Tool-Use""unsloth/gemma-3-27b-pt-bnb-4bit"
        elif model == "llama3_70b_r1":
            model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        elif model == "qwen_70b":
            model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        elif model == "mixtral_70b":
            model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        elif model == "mistral_24b":
            model_id = "mistralai/Mistral-Small-24B-Instruct-2501"
        elif model == "gemma_24b":
            model_id = "google/gemma-3-27b-it"
        cache_path = self.get_cache_path(model_id, "4bit_nf4")
        
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        if self.is_cached(cache_path):
            print(f"Loading cached quantized model from {cache_path}")
            model_nf4 = transformers.AutoModelForCausalLM.from_pretrained(
                str(cache_path),
                quantization_config=nf4_config,
                local_files_only=True,
                device_map=device_map
            )
            tokenizer = AutoTokenizer.from_pretrained(str(cache_path))
        else:
            print(f"Model not cached. Loading and quantizing to {cache_path}")
            model_nf4 = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=nf4_config,
                device_map=device_map,
                cache_dir=str(self.cache_dir),
                token=LLAMA33_ACCESS_TOKEN 
            )
            model_nf4.save_pretrained(str(cache_path))
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                token=LLAMA33_ACCESS_TOKEN,
                cache_dir=str(self.cache_dir)
            )
            tokenizer.save_pretrained(str(cache_path))

        tokenizer.pad_token = tokenizer.eos_token
        self._setup_chat_template(tokenizer)
        return self._create_pipeline(model_nf4, tokenizer)

    def load_8b_model(self, device_map, model):
        """Load 8B model or Phi-4 model."""
        model_id = "aaditya/Llama3-OpenBioLLM-8B"
        if model == "llama3_8b":
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif model == "llama3_70b_2b":
            model_id = "ISTA-DASLab/Meta-Llama-3.1-70B-Instruct-AQLM-PV-2Bit-1x16"
        elif model == "phi4" or model == "phi4_mini":
            model_id = "microsoft/Phi-3-mini-128k-instruct"
        
        cache_path = self.get_cache_path(model_id, "bfloat16")
        
        # Check if we're loading Phi-4 model which requires trust_remote_code, turn this off!
        trust_remote_code = False
        
        if self.is_cached(cache_path):
            print(f"Loading cached model from {cache_path}")
            model_nf4 = transformers.AutoModelForCausalLM.from_pretrained(
                str(cache_path),
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(cache_path),
                trust_remote_code=trust_remote_code
            )
        else:
            print(f"Model not cached. Loading and caching to {cache_path}")
            model_nf4 = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                cache_dir=str(self.cache_dir),
                token=LLAMA31_ACCESS_TOKEN if "llama" in model_id.lower() else None,
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code
            )
            model_nf4.save_pretrained(str(cache_path))
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                token=LLAMA31_ACCESS_TOKEN if "llama" in model_id.lower() else None,
                cache_dir=str(self.cache_dir),
                trust_remote_code=trust_remote_code
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained(str(cache_path))

        self._setup_chat_template(tokenizer)
        
        return self._create_pipeline(model_nf4, tokenizer)

    def _setup_chat_template(self, tokenizer):
        if tokenizer.chat_template is None:
            tokenizer.chat_template = """
            {% for message in messages %}
            {% if message['role'] == 'system' %}
            System: {{ message['content'] }}
            {% elif message['role'] == 'user' %}
            Human: {{ message['content'] }}
            {% elif message['role'] == 'assistant' %}
            Assistant: {{ message['content'] }}
            {% endif %}
            {% endfor %}
            {% if add_generation_prompt %}
            Assistant:
            {% endif %}
            """
    
    def _create_pipeline(self, model, tokenizer, task="text-generation"):
        """Create a pipeline with the given model and tokenizer."""
        if task == "image-text-to-text":
            pipeline_instance = transformers.pipeline(
                task,
                model=model,
                tokenizer=tokenizer,
            )
        else:
            pipeline_instance = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        
        # Test the pipeline
        messages = [
            {"role": "system", "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your job is to annotate different medical contexts and answer related questions. Please answer the below message."},
            {"role": "user", "content": "Hello?"},
        ]
        
        prompt = pipeline_instance.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        terminators = [
            pipeline_instance.tokenizer.eos_token_id,
            pipeline_instance.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = pipeline_instance(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
        print(outputs[0]["generated_text"][len(prompt):])
        return pipeline_instance