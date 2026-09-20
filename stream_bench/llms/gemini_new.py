import json
import os
import time

import requests
from stream_bench.llms.utils import retry_with_exponential_backoff


class GeminiNew:
    def __init__(self, model_name: str = "gemini-2.5-flash-preview-05-20") -> None:
        self.api_key = os.getenv("GCP_API_KEY")
        self.model = model_name
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, max_tokens=1024, temperature=0.0, top_p=1, top_k=1) -> tuple[str, dict]:
        """Returns the tuple of the response text as well as other detailed info."""
        headers = {
            'Content-Type': 'application/json'
        }

        params = {
            'key': self.api_key
        }
        if "gemini-2.5" in self.model:
            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": float(temperature),
                    "maxOutputTokens": int(max_tokens),
                    "thinkingConfig": {
                        "thinkingBudget": 4096
                    },
                    "topP": float(top_p),
                    "responseMimeType": "text/plain"
                }
            }
        else:
            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": float(temperature),
                    "maxOutputTokens": int(max_tokens),
                    "topP": float(top_p),
                    "responseMimeType": "text/plain"
                }
            }

        retry_times = 0
        while retry_times < 3:
            response = requests.post(
                self.base_url,
                headers=headers,
                params=params,
                json=data
            )

            if response.status_code != 200:
                raise requests.exceptions.HTTPError(f"API request failed with status {response.status_code}: {response.text}")

            result = response.json()
            # return result, {}
                # Single response
            if 'candidates' in result and result['candidates']:
                if 'parts' in result['candidates'][0]['content']:
                    res_text = result['candidates'][0]['content']['parts'][0]['text']
                    break
                else:
                    print(f"API return nothing, retry... {retry_times} times")
                    retry_times += 1
                    time.sleep(1)
        while retry_times >= 3 and retry_times < 6:
            data["generationConfig"]['temperature'] = 0.1
            response = requests.post(
                self.base_url,
                headers=headers,
                params=params,
                json=data
            )

            if response.status_code != 200:
                raise requests.exceptions.HTTPError(f"API request failed with status {response.status_code}: {response.text}")

            result = response.json()
            # return result, {}
                # Single response
            if 'candidates' in result and result['candidates']:
                if 'parts' in result['candidates'][0]['content']:
                    res_text = result['candidates'][0]['content']['parts'][0]['text']
                    break
                else:
                    print(f"API return nothing, retry... {retry_times} times")
                    retry_times += 1
                    time.sleep(2)
        while retry_times >= 6 and retry_times < 9:
            data["generationConfig"]['temperature'] = 0.2
            response = requests.post(
                self.base_url,
                headers=headers,
                params=params,
                json=data
            )

            if response.status_code != 200:
                raise requests.exceptions.HTTPError(f"API request failed with status {response.status_code}: {response.text}")

            result = response.json()
            # return result, {}
                # Single response
            if 'candidates' in result and result['candidates']:
                if 'parts' in result['candidates'][0]['content']:
                    res_text = result['candidates'][0]['content']['parts'][0]['text']
                    break
                else:
                    print(f"API return nothing, retry... {retry_times} times")
                    retry_times += 1
                    time.sleep(4)
        while retry_times >= 9:
            raise Exception("API return nothing, retry 10 times, stop the program")
            # res_text = result['candidates'][0]['content']['parts'][0]['text']
        num_input_tokens = result['usageMetadata']['promptTokenCount']
        num_output_tokens = result['usageMetadata']['candidatesTokenCount']
        if 'thoughtsTokenCount' in result['usageMetadata']:
            num_thinking_tokens = result['usageMetadata']['thoughtsTokenCount']
        else:
            num_thinking_tokens = 0

        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "num_thinking_tokens": num_thinking_tokens,
            "logprobs": []  # not provided by this API
        }

        return res_text, res_info


if __name__ == "__main__":
    # Test the GeminiDev class
    print("Testing GeminiDev...")

    # Initialize with default model
    # llm = GeminiNew(model_name="gemini-2.5-flash-preview-05-20")
    llm = GeminiNew(model_name="gemini-2.5-flash-lite-preview-06-17")
    # llm = GeminiNew(model_name="gemma-3-27b-it")
    # llm = GeminiNew(model_name="gemini-2.0-flash-001")

    # Test with a simple prompt
    test_prompt = "Hello! Can you breifly introduce yourself? Please answer in Traditional Chinese."
    print(f"Prompt: {test_prompt}")

    try:
        res_text, res_info = llm(test_prompt)
        print(f"Response: {res_text}")
        print("Test completed successfully!")
        print(res_info)
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Make sure GOOGLE_API_KEY environment variable is set.")
