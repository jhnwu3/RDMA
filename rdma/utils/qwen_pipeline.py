import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Union, Optional, Any
import warnings


class QwenPipeline:
    """
    Simplified custom pipeline class for Qwen models that mimics transformers.pipeline interface.
    Takes only model and tokenizer as arguments, just like transformers.pipeline.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        return_full_text: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate text using the Qwen model with pipeline-like interface.

        Args:
            prompt: Input prompt (string) or messages (list of dicts with 'role' and 'content')
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            eos_token_id: End of sequence token ID(s)
            pad_token_id: Padding token ID
            return_full_text: Whether to return the full text including prompt
            **kwargs: Additional generation parameters

        Returns:
            List of dictionaries with 'generated_text' key
        """
        # Handle different input formats
        if isinstance(prompt, str):
            # Convert string prompt to messages format
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):
            # Already in messages format
            messages = prompt
        else:
            raise ValueError("Prompt must be a string or list of message dictionaries")

        # Apply chat template with thinking mode
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # Qwen-specific parameter
            )
        except TypeError:
            # Fallback for models that don't support enable_thinking parameter
            warnings.warn(
                "Model doesn't support enable_thinking parameter, using default chat template"
            )
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Set up generation parameters
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id or self.tokenizer.pad_token_id,
            **kwargs,
        }

        # Handle eos_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, **generation_kwargs)

        # Extract only the newly generated tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # Parse thinking content (Qwen-specific)
        try:
            # Look for the thinking end token (</think> = 151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
            # Skip thinking content, only return the actual response
            content = self.tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")
        except ValueError:
            # No thinking tokens found, treat all as content
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip(
                "\n"
            )

        # Prepare output in same format as transformers.pipeline
        if return_full_text:
            full_text = text + content
            result = {"generated_text": full_text}
        else:
            result = {"generated_text": content}

        return [result]
