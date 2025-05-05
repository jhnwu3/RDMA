import os
import re
import torch
import nltk
from typing import List, Dict, Union, Optional
from itertools import chain
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

class PhenoGPT:
    """
    Wrapper for the PhenoGPT model that identifies medical phenotypic abnormalities from clinical text.
    """
    
    def __init__(
        self, 
        base_model_path: str = "meta-llama/Llama-2-7b-chat-hf",
        lora_weights_path: str = "/home/johnwu3/projects/rare_disease/workspace/repos/PhenoGPT/model/llama2/llama2_lora_weights",
        load_8bit: bool = True,
        device: str = None,
        hf_api_key: str = None
    ):
        """
        Initialize the PhenoGPT model with the specified base model and fine-tuned weights.
        
        Args:
            base_model_path: Path to the base Llama model
            lora_weights_path: Path to the LoRA weights for fine-tuning
            load_8bit: Whether to load the model in 8-bit precision
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detection)
            hf_api_key: HuggingFace API key for accessing gated models
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set HuggingFace API key if provided
        if hf_api_key:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_api_key
            print("HuggingFace API key set from parameter")
        elif os.environ.get('HF_API_KEY'):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get('HF_API_KEY')
            print("HuggingFace API key set from environment variable")
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
            
        # Load tokenizer
        print(f"Loading tokenizer from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side = "left"
        
        # Load base model
        print(f"Loading base model from {base_model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=load_8bit,
            device_map="auto"
        )
        
        # Load LoRA weights
        print(f"Loading LoRA weights from {lora_weights_path}...")
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_weights_path,
            torch_dtype=torch.float16,
        )
        
        # Set model configuration
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Only set these if they aren't already defined
        if not hasattr(self.model.config, "bos_token_id") or self.model.config.bos_token_id is None:
            self.model.config.bos_token_id = self.tokenizer.bos_token_id if hasattr(self.tokenizer, "bos_token_id") else 1
            
        if not hasattr(self.model.config, "eos_token_id") or self.model.config.eos_token_id is None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else 2
        
        # Set generation configuration
        self.generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.5
        )
        
        # Put model in evaluation mode
        self.model.eval()
        print("Model loaded successfully.")
    
    def generate(
        self, 
        text: str, 
        system_prompt: str = None,
        max_new_tokens: int = 300
    ) -> List[str]:
        """
        Generate phenotypic abnormalities from the input text.
        
        Args:
            text: The clinical text to analyze
            system_prompt: Custom system prompt to override the default
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            List of identified phenotypic abnormalities
        """
        # Default prompt if not specified
        if system_prompt is None:
            system_prompt = "You are a medical assistant and reading a clinical note. Identify a list of all medical phenotypic abnormalities from input text. Format your answer as a list of the phenotypes separated by new line character and do not generate random answers. Only output the list."
        
        # Format the prompt
        base_prompt = """<s>[INST]\n<<SYS>>\nInstructions: {system_prompt}\n<</SYS>>\nInput: {user_prompt}[/INST]\n ### Response: """
        prompt = base_prompt.format(
            system_prompt=system_prompt,
            user_prompt=text
        )
        
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Check for token limit
        if len(input_ids[0]) > 2048:
            print("WARNING: Your text input has more than the predefined maximum 2048 tokens. The results may be defective.")
        
        # Generate output
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        
        # Decode the output
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        
        # Clean the output
        phenotypes = self._clean_output(output)
        
        return phenotypes
    
    def _remove_hpo(self, text: str) -> str:
        """
        Remove HPO IDs from text.
        
        Args:
            text: Text potentially containing HPO IDs
            
        Returns:
            Text with HPO IDs removed
        """
        if "HP:XXXXXXX" in text:
            pattern = "HP:XXXXXXX"
        else:
            pattern = r'HP:.+'
            
        cleaned_text = re.sub(pattern, '', text)
        
        if 'note' in text.lower():
            text = text.lower()
            cleaned_text = re.sub(r'note:.+', '', text)
            
        return cleaned_text
    
    def _clean_output(self, output: str) -> List[str]:
        """
        Clean the model output to extract phenotypic terms.
        
        Args:
            output: Raw model output
            
        Returns:
            List of cleaned phenotypic terms
        """
        if "### Response:" in output:
            output = output.split("### Response:")[-1].split("\n")
            output = [self._remove_hpo(text) for text in output]
            
            if len(output) > 0:
                output_clean = [t.split("|") for t in output]
                output_clean = list(set(chain(*output_clean)))
                output_clean = [re.sub(r'^[\W\d_]+|[\s\W]+$', '', t) for t in output_clean 
                               if not t.strip().startswith("END") and not t.strip() == '</s>']
                output_clean = [re.sub('</s', '', t) for t in output_clean if t and t != "Phenotype"]
            else:
                print("No medical terms were detected")
                output_clean = []
        else:
            print("No medical terms were detected")
            output_clean = []
        
        return output_clean


class PhenoGPTWithHPO(PhenoGPT):
    """
    Extended PhenoGPT with Human Phenotype Ontology (HPO) ID mapping capabilities.
    """
    
    def __init__(
        self,
        base_model_path: str = "meta-llama/Llama-2-7b-chat-hf",
        lora_weights_path: str = "./model/llama2/phenogpt/llama2_lore_weights",
        biosent2vec_path: str = None,
        hpo_database_path: str = "hpo_database.json",
        load_8bit: bool = False,
        device: str = None,
        hf_api_key: str = None
    ):
        """
        Initialize the PhenoGPT model with HPO mapping capabilities.
        
        Args:
            base_model_path: Path to the base Llama model
            lora_weights_path: Path to the LoRA weights for fine-tuning
            biosent2vec_path: Path to the BioSent2Vec model
            hpo_database_path: Path to the HPO database JSON file
            load_8bit: Whether to load the model in 8-bit precision
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detection)
            hf_api_key: HuggingFace API key for accessing gated models
        """
        # Initialize the base PhenoGPT
        super().__init__(base_model_path, lora_weights_path, load_8bit, device, hf_api_key)
        
        self.biosent2vec = None
        self.hpo_database = None
        self.termDB2vec = None
        
        # Load HPO database and BioSent2Vec model if paths are provided
        if biosent2vec_path and os.path.exists(biosent2vec_path) and hpo_database_path and os.path.exists(hpo_database_path):
            self._load_hpo_mapping(biosent2vec_path, hpo_database_path)
    
    def _load_hpo_mapping(self, biosent2vec_path: str, hpo_database_path: str):
        """
        Load HPO mapping resources.
        
        Args:
            biosent2vec_path: Path to the BioSent2Vec model
            hpo_database_path: Path to the HPO database JSON file
        """
        import joblib
        from nltk.corpus import stopwords
        from string import punctuation
        from nltk import word_tokenize
        
        print(f"Loading HPO database from {hpo_database_path}...")
        self.hpo_database = joblib.load(hpo_database_path)
        
        try:
            import sent2vec
            print(f"Loading BioSent2Vec model from {biosent2vec_path}...")
            self.biosent2vec = sent2vec.Sent2vecModel()
            self.biosent2vec.load_model(biosent2vec_path)
            
            # Preprocess and embed HPO terms
            self.stop_words = set(stopwords.words('english'))
            all_terms = list(self.hpo_database.keys())
            all_terms_preprocessed = [self._preprocess_sentence(txt) for txt in all_terms]
            all_terms_vec = self.biosent2vec.embed_sentences(all_terms_preprocessed)
            self.termDB2vec = {k:v for k,v in zip(all_terms, all_terms_vec) if k != 'All'}
            print("HPO mapping resources loaded successfully.")
        except (ImportError, ModuleNotFoundError):
            print("Warning: sent2vec module not found. HPO mapping will not be available.")
            self.biosent2vec = None
            self.termDB2vec = None
    
    def _preprocess_sentence(self, text: str) -> str:
        """
        Preprocess text for BioSent2Vec embedding.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        from nltk import word_tokenize
        from string import punctuation
        
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()

        tokens = [token for token in word_tokenize(text) 
                 if token not in punctuation and token not in self.stop_words]

        return ' '.join(tokens)
    
    def generate_with_hpo(
        self, 
        text: str, 
        system_prompt: str = None,
        max_new_tokens: int = 300
    ) -> Dict[str, str]:
        """
        Generate phenotypic abnormalities with HPO IDs from the input text.
        
        Args:
            text: The clinical text to analyze
            system_prompt: Custom system prompt to override the default
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary mapping phenotypic terms to HPO IDs
        """
        from scipy.spatial import distance
        import numpy as np
        
        # Check if HPO mapping is available
        if self.biosent2vec is None or self.termDB2vec is None or self.hpo_database is None:
            print("Warning: HPO mapping resources not available. Returning phenotypes without HPO IDs.")
            return {term: "N/A" for term in self.generate(text, system_prompt, max_new_tokens)}
        
        # Generate phenotypes
        phenotypes = self.generate(text, system_prompt, max_new_tokens)
        
        # Map to HPO IDs
        all_terms = list(self.termDB2vec.keys())
        all_terms_vec = list(self.termDB2vec.values())
        answers_preprocessed = [self._preprocess_sentence(txt) for txt in phenotypes]
        answer_vec = self.biosent2vec.embed_sentences(answers_preprocessed)
        term2hpo = {}
        
        for i, phenoterm in enumerate(answer_vec):
            all_distances = {}
            dist = []
            
            # Check for exact match first
            if phenotypes[i].capitalize() in all_terms:
                term2hpo[phenotypes[i]] = self.hpo_database[phenotypes[i].capitalize()]
                continue
                
            # Calculate distances to all terms
            for j, ref in enumerate(all_terms_vec):
                dis = distance.cosine(phenoterm, ref)
                if dis > 0:
                    all_distances[all_terms[j]] = 1 - dis
                    dist.append(1-dis)
                    
            # Find closest match
            if len(dist) != 0:
                matched_pheno = list(all_distances.keys())[np.argmax(dist)]
                hpo_id = self.hpo_database[matched_pheno]
                term2hpo[phenotypes[i]] = hpo_id
            else:
                term2hpo[phenotypes[i]] = "N/A"
                
        return term2hpo


# Example usage:
if __name__ == "__main__":
    # Basic PhenoGPT with HuggingFace API key
    # Option 1: Provide API key directly
    # phenogpt = PhenoGPT(
    #     base_model_path="meta-llama/Llama-2-7b-chat-hf",  # You can change this to any compatible model
    #     hf_api_key="your_huggingface_api_key_here"  # Replace with your actual API key
    # )
    
    # Option 2: Use API key from environment variable
    # First set the environment variable: export HF_API_KEY=your_key
    # Then initialize without explicitly passing the key
    phenogpt = PhenoGPT(
        base_model_path="meta-llama/Llama-2-7b-chat-hf"
    )
    
    example_text = """
    The patient is a 45-year-old male presenting with persistent headache, 
    fever, and fatigue for the past week. Physical examination reveals tachycardia,
    elevated blood pressure, and mild jaundice. Laboratory tests show elevated liver enzymes
    and moderate anemia.
    """
    
    phenotypes = phenogpt.generate(example_text)
    print("Detected phenotypes:")
    for p in phenotypes:
        print(f"- {p}")
    
    # With HPO mapping (if BioSent2Vec and HPO database are available)
    # phenogpt_hpo = PhenoGPTWithHPO(
    #     biosent2vec_path="/path/to/BioSentVec_PubMed_MIMICIII-bigram_d700.bin",
    #     hpo_database_path="hpo_database.json",
    #     hf_api_key="your_huggingface_api_key_here"  # Replace with your actual API key
    # )