from datetime import datetime
import time
from utils.llm import ModelLoader
import sys
import json

class Config:
    FLAG_FILE = "process_completed.flag"
    TEMP_FILES = [
        'temp_combined_results.pkl',
        'temp_exact_matches.pkl',
        'temp_non_exact_matches.pkl',
        'temp_final_result.pkl',
        'responses_backup.pkl'
    ]
    MAX_QUERIES_PER_MINUTE = 30
    MAX_TOKENS_PER_DAY = 500000
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
    
    def __init__(self):
        self.prompts = self.load_prompts()
    
    @staticmethod
    def timestamped_print(message):
        """Prints a message with the current timestamp."""
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

    @staticmethod
    def load_prompts(file_path="data/prompts/system_prompts.json"):
        with open(file_path, "r") as file:
            return json.load(file)