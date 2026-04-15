import transformers
import torch
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer
from pathlib import Path
import os
from rdma.utils.api_keys import LLAMA33_ACCESS_TOKEN, LLAMA31_ACCESS_TOKEN
from rdma.utils.qwen_pipeline import (
    QwenPipeline,
)  # Assuming you save the above code as qwen_pipeline.py

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
        model_name = model_id.split("/")[-1]

        cache_path = self.cache_dir / f"{model_name}_{quantization_type}"
        if "gemma" in model_id or "Nemotron" in model_id:
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
        if devices == "auto":
            return "auto"
        elif isinstance(devices, str):
            if devices == "cpu":
                return {"": devices}
            elif "cuda" in devices:
                # For single GPU, use simple mapping
                return {"": devices}
            else:
                raise ValueError(f"Unsupported device specification: {devices}")
        elif isinstance(devices, list):
            if len(devices) == 1:
                return {"": devices[0]}
            else:
                # For multiple GPUs, let HuggingFace handle the distribution
                return "auto"
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
            elif model == "nemotron_120b":
                print("Loading pre-quantized Nemotron 120B NVFP4 model")
                return self.load_nemotron_120b_model(device_map)
            elif (
                "70b" in model.lower()
                or "24b" in model.lower()
                or "32b" in model.lower()
            ):
                print("Loading 70B model with quantization:", model)
                return self.load_70b_model(device_map, model)
            elif (
                "8b" in model.lower()
                or "phi4" in model.lower()
                or "gpt-oss" in model.lower()
            ):  # Modified this line to include phi4 and gpt-oss
                print("Loading 8B model or Phi-4 model or GPT-OSS model:", model)
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
    def load_nuextract_model(
        self, model_name="numind/NuExtract", device="cuda:0", cache_dir=None
    ):
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
                model_name, trust_remote_code=False, cache_dir=cache_dir
            )

            # Load model with appropriate precision
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=False,
                cache_dir=cache_dir,
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
                torch_dtype=torch.bfloat16,
            )
            tokenizer = AutoTokenizer.from_pretrained(str(cache_path))
        else:
            print(f"Model not cached. Loading and caching full model to {cache_path}")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                cache_dir=str(self.cache_dir),
                token=LLAMA33_ACCESS_TOKEN,
                torch_dtype=torch.bfloat16,
            )
            model.save_pretrained(str(cache_path))

            tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=LLAMA33_ACCESS_TOKEN, cache_dir=str(self.cache_dir)
            )
            tokenizer.save_pretrained(str(cache_path))

        self._setup_chat_template(tokenizer)

        return self._create_pipeline(model, tokenizer)

    def load_nemotron_120b_model(self, device_map):
        """Load the pre-quantized NVFP4 Nemotron 120B model."""
        model_id = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
        cache_path = self.get_cache_path(model_id, "nvfp4")  # suffix ignored for Nemotron

        if self.is_cached(cache_path):
            print(f"Loading cached Nemotron 120B model from {cache_path}")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(cache_path),
                device_map=device_map,
                torch_dtype="auto",
                trust_remote_code=True,
                local_files_only=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(cache_path), trust_remote_code=True
            )
        else:
            print(f"Model not cached. Downloading Nemotron 120B to {cache_path}")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                cache_dir=str(self.cache_dir),
                torch_dtype="auto",
                trust_remote_code=True,
            )
            model.save_pretrained(str(cache_path))
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, cache_dir=str(self.cache_dir), trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained(str(cache_path))

        self._setup_chat_template(tokenizer)
        return self._create_pipeline(model, tokenizer)

    def load_gemma_27b_model(self, device_map, model_type="gptq"):
        """Load pre-quantized Gemma 3 27B model."""
        if model_type == "gptq":
            # model_id = "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g"
            model_id = "unsloth/gemma-3-27b-pt-bnb-4bit"
            model_type = ""  # no longer using the gptq
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
                model_id, cache_dir=str(self.cache_dir)
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
            model_id = (
                "Groq/Llama-3-Groq-70B-Tool-Use" "unsloth/gemma-3-27b-pt-bnb-4bit"
            )
        elif model == "llama3_70b_r1":
            model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        elif model == "qwen_70b":
            model_id = "Qwen/Qwen3-32B"
        elif model == "mixtral_70b":
            model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        elif model == "mistral_24b":
            model_id = "mistralai/Mistral-Small-24B-Instruct-2501"
        elif model == "gemma_24b":
            model_id = "google/gemma-3-27b-it"
        elif model == "qwen_32b" or model == "qwen3_32b":
            model_id = "Qwen/Qwen3-32B"
        cache_path = self.get_cache_path(model_id, "4bit_nf4")

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        if self.is_cached(cache_path):
            print(f"Loading cached quantized model from {cache_path}")
            print(device_map)
            print(model_id)
            model_nf4 = transformers.AutoModelForCausalLM.from_pretrained(
                str(cache_path),
                quantization_config=nf4_config,
                local_files_only=True,
                device_map=device_map,
            )
            tokenizer = AutoTokenizer.from_pretrained(str(cache_path))
        else:
            print(f"Model not cached. Loading and quantizing to {cache_path}")
            model_nf4 = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=nf4_config,
                device_map=device_map,
                cache_dir=str(self.cache_dir),
                token=LLAMA33_ACCESS_TOKEN,
            )
            model_nf4.save_pretrained(str(cache_path))

            tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=LLAMA33_ACCESS_TOKEN, cache_dir=str(self.cache_dir)
            )
            tokenizer.save_pretrained(str(cache_path))
        print("Model loaded!")
        tokenizer.pad_token = tokenizer.eos_token
        self._setup_chat_template(tokenizer)
        return self._create_pipeline(model_nf4, tokenizer, model)

    def load_8b_model(self, device_map, model):
        """Load 8B model or Phi-4 model."""
        model_id = "aaditya/Llama3-OpenBioLLM-8B"
        if model == "llama3_8b":
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif model == "llama3_70b_2b":
            model_id = "ISTA-DASLab/Meta-Llama-3.1-70B-Instruct-AQLM-PV-2Bit-1x16"
        elif model == "phi4" or model == "phi4_mini":
            # model_id = "microsoft/Phi-4-mini-128k-instruct"
            model_id = "microsoft/Phi-4-mini-instruct"
        elif model == "gpt-oss":
            model_id = "unsloth/gpt-oss-20b-bnb-4bit"

        cache_path = self.get_cache_path(model_id, "bfloat16")

        # Check if we're loading Phi-4 model which requires trust_remote_code, turn this off!
        trust_remote_code = False
        torch_dtype = torch.bfloat16
        if model == "gpt-oss":
            torch_dtype = "auto"

        if self.is_cached(cache_path):
            print(f"Loading cached model from {cache_path}")
            model_nf4 = transformers.AutoModelForCausalLM.from_pretrained(
                str(cache_path),
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(cache_path), trust_remote_code=trust_remote_code
            )
        else:
            print(f"Model not cached. Loading and caching to {cache_path}")
            model_nf4 = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                cache_dir=str(self.cache_dir),
                token=LLAMA31_ACCESS_TOKEN if "llama" in model_id.lower() else None,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            model_nf4.save_pretrained(str(cache_path))

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=LLAMA31_ACCESS_TOKEN if "llama" in model_id.lower() else None,
                cache_dir=str(self.cache_dir),
                trust_remote_code=trust_remote_code,
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained(str(cache_path))

        self._setup_chat_template(tokenizer)

        return self._create_pipeline(model_nf4, tokenizer, model)

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

    # Replace your existing _create_pipeline method with this updated version
    def _create_pipeline(self, model, tokenizer, model_arg=None):
        """Create a pipeline with the given model and tokenizer."""
        print("Creating pipeline...")
        task = "text-generation"
        # Check if this is a Qwen model that needs special handling
        is_qwen_model = False
        if model_arg and ("qwen" in model_arg.lower() or "gpt" in model_arg.lower()):
            is_qwen_model = True

        if is_qwen_model:
            print("Detected Qwen model, using custom QwenPipeline")
            pipeline_instance = QwenPipeline(model=model, tokenizer=tokenizer)
        else:
            # Use standard transformers pipeline for non-Qwen models
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

        # Test the pipeline (works for both QwenPipeline and transformers.pipeline)
        messages = [
            {
                "role": "system",
                "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your job is to annotate different medical contexts and answer related questions. Please answer the below message.",
            },
            {"role": "user", "content": "Hello?"},
        ]
        try:
            if is_qwen_model:
                # Test QwenPipeline
                outputs = pipeline_instance(
                    prompt=messages,
                    max_new_tokens=50,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                )
                print(
                    "QwenPipeline test output:", outputs[0]["generated_text"][:100000]
                )
            else:
                # Test standard pipeline
                prompt = pipeline_instance.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                terminators = [
                    pipeline_instance.tokenizer.eos_token_id,
                    pipeline_instance.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]

                outputs = pipeline_instance(
                    prompt,
                    max_new_tokens=50,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                )
                print(
                    "pipeline test output:",
                    outputs[0]["generated_text"][len(prompt) :],
                )
        except Exception as e:
            print(f"Pipeline test warning and error: {e}")

        return pipeline_instance
