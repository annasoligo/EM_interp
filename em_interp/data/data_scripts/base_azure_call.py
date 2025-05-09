
from typing import Dict, List
import math
from functools import lru_cache
from pathlib import Path
import yaml
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

endpoint = "https://ft-test-sweden.openai.azure.com/"
deployment = "gpt-4o-N2"

api_version = "2024-12-01-preview"

client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

class AzureRequest:
    """Azure OpenAI models tokenize all numbers from 0-100 as single tokens, which is why we can get exactly 
    one completion token with logprobs. Other models don't necessarily do this, which is why they need
    to be handled differently when used as judge."""
    
    def __init__(self, deployment: str, max_tokens: int = 200, temperature: float = 1):
        self.deployment = deployment
        self.max_tokens = max_tokens
        self.temperature = temperature
        
    
    async def request(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return completion.choices[0].message.content