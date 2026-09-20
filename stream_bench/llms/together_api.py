import os
from typing import Any, Dict, Tuple

import requests
from stream_bench.llms.base import LLM

from .utils import retry_with_exponential_backoff


class TogetherAI(LLM):
    def __init__(self, model_name='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free') -> None:
        self.api_url = 'https://api.together.xyz/v1/chat/completions'
        self.api_key = os.environ.get('TOGETHER_API_KEY')
        if not self.api_key:
            raise ValueError("The TOGETHER_API_KEY environment variable is not set")
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, max_tokens: int = 1024, temperature=0.7, **kwargs) -> Tuple[str, Dict[str, Any]]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature)
        }

        # Pass through the remaining optional parameters
        for key in ['top_p', 'top_k', 'presence_penalty', 'frequency_penalty']:
            if key in kwargs:
                data[key] = kwargs[key]

        response = requests.post(
            self.api_url,
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise ValueError(f"API call failed: {response.status_code}, {response.text}")

        response_data = response.json()
        res_text = response_data['choices'][0]['message']['content']

        # Token usage reported by the Together API
        num_input_tokens = response_data.get('usage', {}).get('prompt_tokens', len(prompt.split()))
        num_output_tokens = response_data.get('usage', {}).get('completion_tokens', len(res_text.split()))

        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "num_thinking_tokens": 0,
            "logprobs": []  # not provided by this API
        }

        return res_text, res_info