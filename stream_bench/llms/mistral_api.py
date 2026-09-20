import json
import os
from typing import Any, Dict, Tuple

import requests
from stream_bench.llms.base import LLM

from .utils import retry_with_exponential_backoff


class MistralAPI(LLM):
    def __init__(self, model_name='mistral-large-latest') -> None:
        self.api_url = 'https://api.mistral.ai/v1/conversations'
        self.api_key = os.environ.get('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("The MISTRAL_API_KEY environment variable is not set")
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, max_tokens: int = 1024, temperature=0.0, **kwargs) -> Tuple[str, Dict[str, Any]]:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        data = {
            "model": self.model_name,
            "inputs": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "completion_args": {
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": 1
            },
            "stream": False
        }

        # Pass through the remaining optional parameters
        for key in ['top_p', 'presence_penalty', 'frequency_penalty']:
            if key in kwargs:
                data[key] = kwargs[key]

        response = requests.post(
            self.api_url,
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise ValueError(f"API request failed: {response.status_code}, {response.text}")

        response_data = response.json()
        res_text = response_data['outputs'][0]['content']

        # The Mistral API reports token usage
        num_input_tokens = response_data['usage'].get('prompt_tokens', len(prompt.split()))  # API-reported value, or an estimate
        num_output_tokens = response_data['usage'].get('completion_tokens', len(res_text.split()))  # API-reported value, or an estimate

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
    # llm = MistralAPI(model_name="mistral-large-latest")
    # llm = MistralAPI(model_name="magistral-medium-2506")
    llm = MistralAPI(model_name="magistral-small-2506")
    # prompt = """Here is the user's requirements for solving a programming problem (enclosed in '''):\n'''\nProblem:\n\nI have a logistic regression model using Pytorch, where my input is high-dimensional and my output must be a scalar - 0, 1 or 2.\n\nI'm using a linear layer combined with a softmax layer to return a n x 3 tensor, where each column represents the probability of the input falling in one of the three classes (0, 1 or 2).\n\nHowever, I must return a 1 x n tensor, and I want to somehow pick the lowest probability for each input and create a tensor indicating which class had the lowest probability. How can I achieve this using Pytorch?\n\nTo illustrate, my Softmax outputs this:\n\n[[0.2, 0.1, 0.7],\n[0.6, 0.3, 0.1],\n[0.15, 0.8, 0.05]]\nAnd I must return this:\n\n[1, 2, 2], which has the type torch.LongTensor\n\n\nA:\n\n<code>\nimport numpy as np\nimport pandas as pd\nimport torch\nsoftmax_output = load_data()\ndef solve(softmax_output):\n</code>\ny = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n'''\n\nYou need to provide your solution in python code to satisfy the user's requirements. Your code will be tested as follows (enclosed in '''):\n'''\n['exec_context = r\"\"\"\\nimport numpy as np\\nimport pandas as pd\\nimport torch\\nsoftmax_output = test_input\\ndef solve(softmax_output):\\n[insert]\\n    return y\\ny = solve(softmax_output)\\nresult = y\\n\"\"\"']\n\ncode = exec_context.replace(\"[insert]\", <your_code>)\na_test_case = generate_test_case()\ntest_input, expected_result = a_test_case\ntest_env = {\"test_input\": test_input}\nexec(code, test_env)\nassertEqual(test_env[\"result\"], expected_result)\n'''\n\nNow, generate your code directly in the following format:\n```python\n<your_code>\n```", "output_pred": "```python\ny = torch.argmin(softmax_output, dim=1)\n```"""
    prompt = "Which country is famously known for its tulips? A) Italy B) France C) Netherlands D) Germany? Only answer with the letter"
    res_text, res_info = llm(prompt, 16384)
    print(res_text)
    # print(res_info)
    print(res_info['num_input_tokens'])
    print(res_info['num_output_tokens'])