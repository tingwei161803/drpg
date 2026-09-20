import json
import os
from typing import Any, Dict, Tuple

import requests
from stream_bench.llms.base import LLM

from .utils import retry_with_exponential_backoff


class NvidiaAPI(LLM):
    def __init__(self, model_name='meta/llama-3.3-70b-instruct') -> None:
        self.invoke_url = 'https://integrate.api.nvidia.com/v1/chat/completions'
        self.api_key = os.environ.get('NVIDIA_API_KEY')
        if not self.api_key:
            raise ValueError("The NVIDIA_API_KEY environment variable is not set")
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, max_tokens: int = 1024, temperature=0.2, **kwargs) -> Tuple[str, Dict[str, Any]]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "model": self.model_name,
            "max_tokens": int(max_tokens),
            "presence_penalty": kwargs.get('presence_penalty', 0),
            "frequency_penalty": kwargs.get('frequency_penalty', 0),
            "top_p": kwargs.get('top_p', 0.7),
            "temperature": float(temperature)
        }

        # Only genuine reasoning models run with thinking ON. Newer qwen hybrids
        # (qwen3.5-*, qwen3-next-*-instruct) must run WITHOUT thinking, so force it
        # off explicitly (some default to thinking ON otherwise).
        THINKING_MODELS = ('qwq', 'deepseek', 'granite', 'qwen3-235b')
        if any(t in self.model_name for t in THINKING_MODELS):
            data['chat_template_kwargs'] = {"thinking": True}
        elif 'qwen' in self.model_name:
            data['chat_template_kwargs'] = {"thinking": False}

        response = requests.post(
            self.invoke_url,
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise ValueError(f"API request failed: {response.status_code}, {response.text}")

        response_data = response.json()
        res_text = response_data['choices'][0]['message']['content']

        # This API does not return logprobs, so we fill in a simplified response
        num_input_tokens = response_data['usage']['prompt_tokens']
        num_output_tokens = response_data['usage']['completion_tokens']

        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "num_thinking_tokens": 0,
            "logprobs": []  # not provided by this API
        }

        return res_text, res_info