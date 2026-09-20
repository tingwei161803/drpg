import json
import os
from typing import Any, Dict, Tuple

import requests
from stream_bench.llms.base import LLM

from .utils import retry_with_exponential_backoff


class LlamaAPI(LLM):
    def __init__(self, model_name='Llama-3.3-70B-Instruct') -> None:
        self.invoke_url = 'https://api.llama.com/v1/chat/completions'
        self.api_key = os.environ.get('LLAMA_API_KEY')
        if not self.api_key:
            raise ValueError("The LLAMA_API_KEY environment variable is not set")
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
            # "presence_penalty": kwargs.get('presence_penalty', 0),
            # "frequency_penalty": kwargs.get('frequency_penalty', 0),
            "top_p": kwargs.get('top_p', 0.7),
            "temperature": float(temperature)
        }

        # if 'qwq' in self.model_name or 'qwen' in self.model_name or 'deepseek' in self.model_name or 'granite' in self.model_name:
        #     data['chat_template_kwargs'] = {
        #         "thinking": True
        #     }

        response = requests.post(
            self.invoke_url,
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise ValueError(f"API request failed: {response.status_code}, {response.text}")

        response_data = response.json()
        # return response_data, {}
        res_text = response_data['completion_message']['content']['text']

        # This API does not return logprobs, so we fill in a simplified response
        metrics = {item['metric']: item['value'] for item in response_data['metrics']}
        num_input_tokens = metrics['num_prompt_tokens']
        num_output_tokens = metrics['num_completion_tokens']

        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "num_thinking_tokens": 0,
            "logprobs": []  # not provided by this API
        }

        return res_text, res_info

if __name__ == "__main__":
    llm = LlamaAPI(model_name="Llama-3.3-8B-Instruct")
    prompt = "What is the capital of France?"
    res_text, res_info = llm(prompt, 1024)
    print(res_text)
    print(res_info)