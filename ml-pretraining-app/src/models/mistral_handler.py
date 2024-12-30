import requests
import json
from typing import List, Dict
import numpy as np

class MistralHandler:
    def __init__(self, config):
        self.api_url = config['lmstudio']['api_url']
        self.timeout = config['lmstudio']['timeout']
        
    def process_batch(self, texts: List[str]) -> List[Dict]:
        """
        Traite un lot de textes avec Mistral via LM Studio
        """
        formatted_prompts = [
            {
                "role": "user",
                "content": f"Process this text: {text}"
            } for text in texts
        ]
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json={
                    "messages": formatted_prompts,
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=self.timeout
            )
            return response.json()
        except Exception as e:
            print(f"Error processing batch: {e}")
            return None
