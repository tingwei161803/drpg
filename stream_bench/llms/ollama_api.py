import json
import os
from typing import Any, Dict, Tuple

import requests
from stream_bench.llms.base import LLM

from .utils import retry_with_exponential_backoff


class OllamaAPI(LLM):
    def __init__(self, model_name='llama2') -> None:
        self.api_url = 'http://localhost:11434/v1/chat/completions'
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, max_tokens: int = 1024, temperature=0.0, **kwargs) -> Tuple[str, Dict[str, Any]]:
        headers = {
            'Content-Type': 'application/json'
        }

        messages = [{"role": "user", "content": prompt}]

        # Check whether a system message was provided
        system_message = kwargs.get('system_message', "You are a helpful assistant.")
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature)
        }

        # Pass through the remaining optional parameters
        for key in ['top_p', 'top_k', 'presence_penalty', 'frequency_penalty']:
            if key in kwargs:
                data[key] = kwargs[key]

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data
            )

            if response.status_code != 200:
                raise ValueError(f"API request failed: {response.status_code}, {response.text}")

            response_data = response.json()
            res_text = response_data['choices'][0]['message']['content']

            # The Ollama API may not report complete token usage
            num_input_tokens = len(prompt.split())  # approximation
            num_output_tokens = len(res_text.split())  # approximation

            res_info = {
                "input": prompt,
                "output": res_text,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "num_thinking_tokens": 0,
                "logprobs": []  # not provided by this API
            }

            return res_text, res_info

        except requests.exceptions.ConnectionError:
            raise ConnectionError("Could not reach the Ollama server. Make sure Ollama is running locally.")