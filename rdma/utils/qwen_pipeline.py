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

    def _apply_chat_template(
        self, messages: List[Dict[str, str]], enable_thinking: bool = True
    ) -> str:
        """Apply chat template with proper thinking mode handling"""
        # Check if the last message has thinking control tags
        last_message = messages[-1]["content"]

        # Handle thinking control tags
        if "/no_think" in last_message:
            # Remove the tag and disable thinking
            messages[-1]["content"] = last_message.replace("/no_think", "").strip()
            enable_thinking = False
        elif "/think" in last_message:
            # Remove the tag and enable thinking
            messages[-1]["content"] = last_message.replace("/think", "").strip()
            enable_thinking = True

        # Apply the chat template with thinking mode
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=None,
                add_thinking_prompt=enable_thinking,  # This enables/disables thinking mode
            )
        else:
            # Fallback for tokenizers without apply_chat_template
            text = ""
            for msg in messages:
                text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            text += "<|im_start|>assistant\n"
            if enable_thinking:
                text += "<|thinking|>\n"
            return text

    def __call__(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        do_sample: bool = True,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        return_full_text: bool = False,
        enable_thinking: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate text using the Qwen model with pipeline-like interface.

        Args:
            prompt: Input prompt (string) or messages (list of dicts with 'role' and 'content')
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.6 for thinking mode, 0.7 for non-thinking)
            top_p: Top-p sampling parameter (0.95 for thinking mode, 0.8 for non-thinking)
            top_k: Top-k sampling parameter (default 20)
            do_sample: Whether to use sampling (should be True, not greedy decoding)
            eos_token_id: End of sequence token ID(s)
            pad_token_id: Padding token ID
            return_full_text: Whether to return the full text including prompt
            enable_thinking: Whether to enable Qwen3 thinking mode
            **kwargs: Additional generation parameters

        Returns:
            List of dictionaries with 'generated_text' key
        """
        # Handle different input formats
        if isinstance(prompt, str):
            # Check if this is already a pre-formatted chat template
            if any(
                marker in prompt
                for marker in [
                    "<|im_start|>",
                    "<|begin_of_text|>",
                    "system:",
                    "user:",
                    "assistant:",
                ]
            ):
                # This looks like a pre-formatted chat template, use it as-is
                print("DEBUG: Using pre-formatted chat template directly")
                text = prompt
            else:
                # Convert string prompt to messages format
                messages = [{"role": "user", "content": prompt}]
                text = self._apply_chat_template(messages, enable_thinking)
        elif isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):
            # Already in messages format
            text = self._apply_chat_template(prompt, enable_thinking)
        else:
            raise ValueError("Prompt must be a string or list of message dictionaries")

        # Tokenize input - Fix: Use proper device handling
        inputs = self.tokenizer(text, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)

        # Set up generation parameters
        generation_params = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id or self.tokenizer.pad_token_id,
        }

        if do_sample:
            generation_params.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )
        else:
            generation_params["do_sample"] = False

        print(f"DEBUG: Generation parameters: {generation_params}")

        # Use Qwen's proper EOS tokens so generation stops naturally
        generation_params["eos_token_id"] = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ]

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_params)

            # Extract only newly generated token IDs
            output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

            # Strip thinking block using token ID 151668 (</think>), per HF Qwen3 docs
            think_end_token = self.tokenizer.convert_tokens_to_ids("</think>")
            try:
                index = len(output_ids) - output_ids[::-1].index(think_end_token)
            except ValueError:
                index = 0  # no thinking block present

            content = self.tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip()

            if return_full_text:
                result = {"generated_text": text + content}
            else:
                result = {"generated_text": content}

            return [result]

        except Exception as e:
            print(f"Generation error: {e}")
            return [{"generated_text": ""}]


# Example usage to demonstrate the fixed pipeline
if __name__ == "__main__":
    # Load model and tokenizer
    model_name = "Qwen/Qwen3-32B"  # or whatever model you're using
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create pipeline
    pipeline = QwenPipeline(model, tokenizer)

    # Test with different formats
    print("=== Test 1: String prompt ===")
    result = pipeline("How many r's in strawberries?", max_new_tokens=1000)
    print(f"Result: {result[0]['generated_text']}")

    print("\n=== Test 2: Messages format ===")
    messages = [{"role": "user", "content": "How many r's in strawberries?"}]
    result = pipeline(messages, max_new_tokens=1000)
    print(f"Result: {result[0]['generated_text']}")

    print("\n=== Test 3: With /no_think tag ===")
    result = pipeline("How many r's in blueberries? /no_think", max_new_tokens=1000)
    print(f"Result: {result[0]['generated_text']}")
