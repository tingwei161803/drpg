import time
import urllib.error
import urllib.request

import requests
from colorama import Fore, Style


def _provider_errors() -> tuple:
    """Collect the retryable exception types of whichever provider SDKs are installed.

    Every backend in ``stream_bench/llms`` only talks to a single provider, so the
    imports below are optional on purpose: running DRPG on, say, NVIDIA NIM should
    not require installing the OpenAI, Anthropic, Groq and Together clients as well.
    Whatever is installed is retried exactly as in the original StreamBench code.
    """
    errors: list = []
    try:
        import openai
        errors += [openai.RateLimitError, openai.APIError]
    except ImportError:
        pass
    try:
        import google.api_core.exceptions as g_exceptions
        errors += [
            g_exceptions.ResourceExhausted,
            g_exceptions.ServiceUnavailable,
            g_exceptions.GoogleAPIError,
        ]
    except ImportError:
        pass
    try:
        import anthropic
        errors += [
            anthropic.BadRequestError,
            anthropic.InternalServerError,
            anthropic.RateLimitError,
        ]
    except ImportError:
        pass
    try:
        import groq
        errors += [groq.RateLimitError, groq.InternalServerError, groq.APIConnectionError]
    except ImportError:
        pass
    try:
        import together
        errors += [together.error.TogetherException]
    except ImportError:
        pass
    return tuple(errors)


PROVIDER_ERRORS = _provider_errors()


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 4,
    exponential_base: float = 1.5,
    max_retries: int = 10
):
    # Define errors based on available libraries.
    errors_tuple = PROVIDER_ERRORS + (
        urllib.error.HTTPError, urllib.error.URLError,
        requests.exceptions.RequestException, requests.exceptions.ConnectionError,
        requests.exceptions.Timeout, requests.exceptions.HTTPError,
        ValueError, IndexError, UnboundLocalError
    )
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specific errors
            except errors_tuple as e:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if isinstance(e, ValueError) or (num_retries > max_retries):
                    print(Fore.RED + f"ValueError / Maximum number of retries ({max_retries}) exceeded." + Style.RESET_ALL)
                    result = 'error:{}'.format(e)
                    prompt = kwargs["prompt"] if "prompt" in kwargs else args[1]
                    res_info = {
                        "input": prompt,
                        "output": result,
                        "num_input_tokens": len(prompt) // 4,  # approximation
                        "num_output_tokens": 0,
                        "num_thinking_tokens": 0,
                        "logprobs": []
                    }
                    return result, res_info
                # Sleep for the delay
                print(Fore.YELLOW + f"Error encountered ({e}). Retry ({num_retries}) after {delay} seconds..." + Style.RESET_ALL)
                time.sleep(delay)
                # Increment the delay
                delay *= exponential_base
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
    return wrapper
