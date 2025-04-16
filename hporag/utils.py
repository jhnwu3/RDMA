# utils.py
import re
import string
import json
from fuzzywuzzy import fuzz

class TextUtils:
    @staticmethod
    def clean_text(text):
        """Cleans input text by converting to lowercase and removing punctuation."""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation)).strip()
        return text

    @staticmethod
    def clean_and_parse(json_str):
        """Cleans and parses a JSON string by fixing formatting issues."""
        try:
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'\s+', ' ', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def process_unique_metadata(metadata):
        """Processes unique metadata by converting all keys to lowercase."""
        if not isinstance(metadata, list):
            return []
            
        processed_list = []
        for item in metadata:
            try:
                item_dict = json.loads(item)
                processed_item = {k.lower(): v for k, v in item_dict.items()}
                processed_list.append(json.dumps(processed_item))
            except (json.JSONDecodeError, TypeError):
                continue
        return processed_list