import re
from enum import Enum

from stream_bench.agents.base import Agent
from stream_bench.agents.utils_rag import RAG


class Method(Enum):
    POSTHOC_POLICY = "posthoc_policy"
    RAG_POLICY = "rag_policy_rag"


class PolicyRAGAgent(Agent):
    """Policy agent that uses RAG for few-shot learning and policy generation.

    This agent implements several methods:
    1. Post-hoc policy generation
    2. RAG-based policy generation
    """
    LOG_KEYS_FOR_PROMPT = [
        "input_pred",
        "output_pred",
        "logprobs_pred",
        "input_parse",
        "output_parse",
        "logprobs_parse",
        "input_savedecision",
        "output_savedecision",
        "logprobs_savedecision"
    ]
    METHODS = {member.value for member in Method}

    def __init__(self, config: dict) -> None:
        """Initialize the PolicyAgent.

        Args:
            config: Configuration dictionary for the policy agent.
        """
        # Validate and set agent name
        assert config["agent_name"] in self.METHODS, (
            f"Agent name must be one of {self.METHODS}")
        self.method = config["agent_name"]

        # Set mode
        assert config["mode"] in {"only_correct", "only_incorrect", "normal"}
        self.mode = config["mode"]

        # Set seed
        self.seed = config["seed"]

        # Set feature flags
        self.use_correct_fewshots = config["use_correct_fewshots"]
        self.enabled_optimization = config["optimization"]
        self.policy_generator_mode = config["policy_generator_mode"]

        # Set enabled_cot parameter (defaults to False if not present)
        self.enabled_cot = config.get("enabled_cot", False)

        # Set policy generation parameters
        self.use_previous_policy = config["use_previous_policy"]
        self.prompt_ver = config["prompt_ver"]

        # set policy generator fewshots num
        self.policy_generator_fewshots_num = config["policy_generator_fewshots_num"]

        # Set enable_transfer and enable_teacher parameters
        # (default to False if not present)
        self.enabled_transfer = config.get("enabled_transfer", False)
        self.enabled_teacher = config.get("enabled_teacher", False)

        # Initialize base agent (after all attributes are set)
        super().__init__(config)

        # Initialize LLM config
        self.llm_config = config["llm"]

        # Initialize policy LLM config
        self.policy_llm_config = {
            "max_tokens": config["policy_llm"]["max_tokens"],
            "model_name": config["policy_llm"]["model_name"],
            "series": config["policy_llm"]["series"],
            "temperature": config["policy_llm"]["temperature"]
        }

        # Initialize RAG
        if config["rag"]["rag_filename"] is None:
            config["rag"]["rag_filename"] = self.log_path + ".db"
        self.rag = RAG(config["rag"])

        # Initialize EWMA
        self.ewma_alpha = config["ewma"]["initial_alpha"]
        self.ewma_value = config["ewma"]["initial_ewma"]
        self.confidence_score = self.ewma_value

        # Initialize counters and state
        self.update_cnt = 0
        self.cur_rationale = None
        self.current_policy = None

        # Store all configuration parameters
        self.config = config
        self.last_case_correctness = None
        self.last_case = None

    def switch_models(self) -> None:
        """Switch from primary models to secondary models."""
        if self.llm2 is not None:
            print("Switching from primary LLM to secondary LLM")
            self.llm = self.llm2
            self.llm_config = self.llm2_config

        if self.policy_llm2 is not None:
            print("Switching from primary policy LLM to secondary policy LLM")
            self.policy_llm = self.policy_llm2
            self.policy_llm_config = self.policy_llm2_config

    def generate_rag_policy(self, y_shots: list[str], n_shots: list[str], **kwargs
                           ) -> tuple[str, dict, str, str]:
        """Generate a policy based on the retrieved shots.

        Args:
            y_shots: List of correct shots
            n_shots: List of incorrect shots

        Returns:
            tuple[str, dict, str, str]: Generated policy, token information,
            policy prompt, policy raw output
        """
        # Select the appropriate policy generator template based on configuration
        template_name = (
            f"rag_policy_{'with' if self.use_previous_policy else 'no'}_prev_policy_"
            f"with_{self.policy_generator_mode}_cases_rule_ver"
            f"{'1' if self.prompt_ver == 'original' else 'LEAP'}"
        )
        template = kwargs.get(template_name)

        # Prepare format parameters based on prompt version and mode
        format_params = {}

        # Add previous policy if using previous policy
        if self.use_previous_policy:
            format_params["previous_policy"] = (
                self.current_policy if self.current_policy else "No previous policy"
            )

        # Add correct cases if mode is "both" or "correct"
        if self.policy_generator_mode in ["both", "correct"]:
            format_params["previous_correct_cases"] = "\n\n".join(y_shots)

        if self.policy_generator_mode in ["both", "wrong"]:
            format_params["previous_wrong_cases"] = "\n\n".join(n_shots)

        # Format the template with the necessary information
        prompt = template.format(**format_params)

        # Generate the policy using the policy LLM
        policy_text, policy_info = self.policy_llm(
            prompt=prompt,
            max_tokens=self.policy_llm_config["max_tokens"],
            temperature=self.policy_llm_config["temperature"]
        )

        # Extract the policy from the response
        policy_match = re.search(
            r"POLICY:\s*(.*)",
            policy_text,
            re.DOTALL
        )
        if policy_match:
            current_policy = policy_match.group(1).strip()
        else:
            current_policy = policy_text.strip()

        return current_policy, policy_info, prompt, policy_text

    def generate_posthoc_policy(self, y_shots: list[str], **kwargs
                               ) -> tuple[str, dict, str, str]:
        """Generate a policy based on the last case.

        Args:
            y_shots: List of correct shots

        Returns:
            tuple[str, dict, str, str]: Generated policy, token information,
            policy prompt, policy raw output
        """
        # If last case was correct, no need to generate new policy
        if self.last_case_correctness:
            return (
                self.current_policy,
                {"num_input_tokens": 0, "num_output_tokens": 0}, "", ""
            )
        # Select the appropriate policy generator template based on configuration
        template_name = (
            f"posthoc_policy_{'with' if self.use_previous_policy else 'no'}_prev_policy_"
            f"with_{self.policy_generator_mode}_cases_rule_ver"
            f"{'1' if self.prompt_ver == 'original' else 'LEAP'}"
        )
        template = kwargs.get(template_name)

        # Prepare format parameters based on prompt version and mode
        format_params = {}

        # Add previous policy if using previous policy
        if self.use_previous_policy:
            format_params["previous_policy"] = (
                self.current_policy if self.current_policy else "No previous policy"
            )

        # Add correct cases if mode is "both" or "correct"
        if self.policy_generator_mode in ["both", "correct"]:
            format_params["previous_correct_cases"] = "\n\n".join(y_shots)

        if self.policy_generator_mode in ["both", "wrong"]:
            format_params["new_wrong_case"] = self.last_case

        # Format the template with the necessary information
        prompt = template.format(**format_params)

        # Generate the policy using the policy LLM
        policy_text, policy_info = self.policy_llm(
            prompt=prompt,
            max_tokens=self.policy_llm_config["max_tokens"],
            temperature=self.policy_llm_config["temperature"]
        )

        # Extract the policy from the response
        policy_match = re.search(
            r"POLICY:\s*(.*)",
            policy_text,
            re.DOTALL
        )
        if policy_match:
            current_policy = policy_match.group(1).strip()
        else:
            current_policy = policy_text.strip()

        return current_policy, policy_info, prompt, policy_text

    def __call__(
        self,
        question: str,
        prompt_zeroshot: str,
        fewshot_template: str,
        prompt_cot: str,
        fewshotcot_template: str = None,
        parse_template: str = None,
        label_set: set[str] = None,
        **kwargs
    ) -> str:
        if label_set is not None:
            assert parse_template is not None
        # Retrieve few-shot examples
        y_shots_original = None
        n_shots_original = None
        if self.rag.insert_acc > 0:
            if self.mode == "normal":
                y_shots_original, n_shots_original = self.rag.retrieve_by_correctness(query=question, top_k=self.rag.top_k)
                if self.rag.retrieve_order == "similar_at_bottom":
                    y_shots = list(reversed(y_shots_original))
                    n_shots = list(reversed(n_shots_original))
                else:
                    y_shots = y_shots_original
                    n_shots = n_shots_original
            else:
                shots = self.rag.retrieve(query=question, top_k=self.rag.top_k)
                y_shots = list()
                n_shots = list()
                not_corr_verb = "not correct"
                for shot in shots:
                    if shot.strip().endswith(not_corr_verb):
                        n_shots.append(shot)
                    else:
                        y_shots.append(shot)
        else:
            y_shots = []
            n_shots = []
        shots = y_shots + n_shots

        # --- Policy retrieval logic ---
        # Try to get policy from most similar correct case
        retrieved_policy = ""
        if y_shots_original and len(y_shots_original) > 0:
            # Assume the first y_shots_original[0] is the most similar correct case
            # Extract the question part from the chunk (assume memprompt_template has question: ...)
            first_correct_chunk = y_shots_original[0]
            # Try to extract the question string
            match = re.search(r"question:(.*?)(?:\\n|$)", first_correct_chunk, re.IGNORECASE)
            if match:
                similar_question = match.group(1).strip()
                retrieved_policy = self.rag.get_policy_by_key(similar_question)
        # If not found or empty, will generate a new policy below

        # Select the appropriate policy generator template based on configuration
        template_w_rag = fewshotcot_template
        template_wo_rag = prompt_cot
        template_w_rag_with_policy = kwargs.get("fewshotcot_template_with_policy")
        template_wo_rag_with_policy = kwargs.get("prompt_cot_with_policy")
        # fewshots_limit = self.rag.top_k//2 if self.policy_generator_fewshots_num == "half_topk" else 2
        if self.policy_generator_fewshots_num == "half_topk":
            if self.policy_generator_mode == "both":
                fewshots_limit = self.rag.top_k // 2
            else:
                fewshots_limit = self.rag.top_k
        elif self.policy_generator_fewshots_num == "two":
            fewshots_limit = 2
        elif self.policy_generator_fewshots_num == "one":
            fewshots_limit = 1
        else:
            fewshots_limit = 1

        if y_shots_original is not None:
            y_shots_for_policy = list(reversed(y_shots_original[:fewshots_limit])) if self.rag.retrieve_order == "similar_at_bottom" else y_shots_original[:fewshots_limit]
        else:
            y_shots_for_policy = []
        if n_shots_original is not None:
            n_shots_for_policy = list(reversed(n_shots_original[:fewshots_limit])) if self.rag.retrieve_order == "similar_at_bottom" else n_shots_original[:fewshots_limit]
        else:
            n_shots_for_policy = []


        # When generating policy, only generate if retrieved_policy is empty
        policy_info = None
        policy_prompt = ""
        policy_raw_output = ""
        if len(shots) > 0 and self.use_correct_fewshots:
            if not retrieved_policy:
                if self.method == Method.RAG_POLICY.value:
                    self.current_policy, policy_info, policy_prompt, policy_raw_output = (
                        self.generate_rag_policy(
                            y_shots=y_shots_for_policy,
                            n_shots=n_shots_for_policy,
                            **kwargs
                        )
                    )
                elif self.method == Method.POSTHOC_POLICY.value:
                    self.current_policy, policy_info, policy_prompt, policy_raw_output = (
                        self.generate_posthoc_policy(
                            y_shots=y_shots_for_policy,
                            **kwargs
                        )
                    )
            else:
                self.current_policy = retrieved_policy

            if self.current_policy is not None:
                fewshot_text = "\n\n\n".join(y_shots).replace("\\", "\\\\")
                try:
                    prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=template_w_rag_with_policy)
                    prompt = re.sub(pattern=r"\{policy\}", repl=self.current_policy, string=prompt)
                except Exception as e:
                    error_msg = f"Error ```{e}``` caused by these shots. Logged to jsonl."
                    print(error_msg)
                    shots.append(error_msg)  # For analyzing errors afterwards
                    prompt = template_wo_rag
            else:
                fewshot_text = "\n\n\n".join(y_shots).replace("\\", "\\\\")
                try:
                    prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=template_w_rag)
                except Exception as e:
                    error_msg = f"Error ```{e}``` caused by these shots. Logged to jsonl."
                    print(error_msg)
                    y_shots.append(error_msg)  # For analyzing errors afterwards
                    prompt = template_wo_rag
        elif len(shots) > 0 and not self.use_correct_fewshots:

            if self.method == Method.RAG_POLICY.value:
                self.current_policy, policy_info, policy_prompt, policy_raw_output = (
                    self.generate_rag_policy(
                        y_shots=y_shots_for_policy,
                        n_shots=n_shots_for_policy,
                        **kwargs
                    )
                )
            elif self.method == Method.POSTHOC_POLICY.value:
                self.current_policy, policy_info, policy_prompt, policy_raw_output = (
                    self.generate_posthoc_policy(
                        y_shots=y_shots_for_policy,
                        **kwargs
                    )
                )
            if self.current_policy is not None:
                try:
                    prompt = re.sub(pattern=r"\{policy\}", repl=self.current_policy, string=template_wo_rag_with_policy)
                except Exception as e:
                    error_msg = f"Error ```{e}``` caused by these shots. Logged to jsonl."
                    print(error_msg)
                    shots.append(error_msg)  # For analyzing errors afterwards
                    prompt = template_wo_rag
            else:
                prompt = template_wo_rag
        else:
            if self.current_policy is not None:
                prompt = template_wo_rag_with_policy
                prompt = re.sub(pattern=r"\{policy\}", repl=self.current_policy, string=prompt)
            else:
                prompt = template_wo_rag
        # Inference
        pred_text_raw, pred_info = self.llm(
            prompt=prompt,
            max_tokens=self.llm_config["max_tokens"],
            temperature=self.llm_config["temperature"]
        )

        # Filter pred_text based on conditions
        pred_text = pred_text_raw

        # Condition 1: If contains </think>, output everything after </think>
        if "</think>" in pred_text_raw:
            think_index = pred_text_raw.find("</think>")
            pred_text = pred_text_raw[think_index + len("</think>"):].strip()
        # Condition 2: If contains ANSWER:, output everything after ANSWER:
        elif "ANSWER:" in pred_text_raw:
            answer_index = pred_text_raw.find("ANSWER:")
            pred_text = pred_text_raw[answer_index + len("ANSWER:"):].strip()

        # logging
        if (self.method in {Method.RAG_POLICY.value, Method.POSTHOC_POLICY.value}
            and policy_info is not None):
            self.update_log_info(log_data=sanitize_log_data({
                "num_shots": str(len(shots)),
                "num_correct_shots": str(len(y_shots)),
                "num_incorrect_shots": str(len(n_shots)),
                "retrieved_shots": shots,
                "correct_shots": y_shots,
                "incorrect_shots": n_shots,
                "input_pred": prompt,
                "output_pred": pred_text,
                "output_pred_raw": pred_text_raw,
                "logprobs_pred": pred_info["logprobs"],
                "num_inference_call": 1,
                "num_success_call": 1 if (pred_text[:5] != "error") else 0,
                "num_input_tokens": (pred_info["num_input_tokens"] +
                                   policy_info["num_input_tokens"]),
                "num_output_tokens": (pred_info["num_output_tokens"] +
                                    policy_info["num_output_tokens"]),
                "num_thinking_tokens": (pred_info.get("num_thinking_tokens", 0) + policy_info.get("num_thinking_tokens", 0)),
                "current_policy": str(self.current_policy),
                "confidence_score": self.confidence_score,
                "policy_prompt": policy_prompt,
                "policy_raw_output": policy_raw_output
            }))
        else:
            self.update_log_info(log_data=sanitize_log_data({
                "num_shots": str(len(shots)),
                "num_correct_shots": str(len(y_shots)),
                "num_incorrect_shots": str(len(n_shots)),
                "retrieved_shots": shots,
                "correct_shots": y_shots,
                "incorrect_shots": n_shots,
                "input_pred": prompt,
                "output_pred": pred_text,
                "output_pred_raw": pred_text_raw,
                "logprobs_pred": pred_info["logprobs"],
                "num_inference_call": 1,
                "num_success_call": 1 if (pred_text[:5] != "error") else 0,
                "num_input_tokens": pred_info["num_input_tokens"],
                "num_output_tokens": pred_info["num_output_tokens"],
                "num_thinking_tokens": pred_info.get("num_thinking_tokens", 0),
                "current_policy": str(self.current_policy),
                "confidence_score": self.confidence_score,
                "policy_prompt": "",
                "policy_raw_output": ""
            }))
        return pred_text

    def update(self, has_feedback: bool, **feedbacks) -> bool:
        question = feedbacks["question"]
        if self.method in {Method.POSTHOC_POLICY.value, Method.RAG_POLICY.value}:
            assert ("self_output" in feedbacks) and ("is_correct" in feedbacks)
            answer = feedbacks["self_output"]
            if feedbacks["is_correct"]:
                corr_verb = "correct"
                self.confidence_score = round(
                    self.ewma_alpha * 1 + (1 - self.ewma_alpha) * self.confidence_score, 2)
                self.last_case_correctness = True
            else:
                corr_verb = "not correct"
                self.confidence_score = round(
                    self.ewma_alpha * 0 + (1 - self.ewma_alpha) * self.confidence_score, 2)
                self.last_case_correctness = False
            correctness_text = f"Your answer is {corr_verb}"
        else:
            raise NotImplementedError

        chunk = feedbacks["memprompt_template"].format(
            question=question, answer=answer, correctness=correctness_text)
        # Only store policy if correct, else store empty string
        if corr_verb == "correct":
            policy = self.current_policy if hasattr(self, "current_policy") else ""
        else:
            policy = ""
        if (self.bench.DATASET_PATH == "hotpot_qa") and (self.bench.DATASET_NAME == "distractor"):
            index = question.index("Question:")
            question = question[index:].strip()
            if self.update_cnt == 0:
                print("Embed question using only the question itself.")
        self.last_case = chunk
        self.rag.insert(key=question, value=chunk, policy=policy)
        self.update_cnt += 1
        return True

    def get_name(self) -> str:
        return "__".join([
            f'{self.config["agent_name"]}-{self.config["mode"]}',
            self.llm_config["series"],
            self.llm_config["model_name"],
            self.config["rag"]["embedding_model"].split('/')[-1],
            self.config["rag"]["order"],
            str(self.config["rag"]["top_k"]),
            f"seed-{self.config['seed']}",
            f"use_corICL-{self.config['use_correct_fewshots']}",
            f"use_prev_policy-{self.config['use_previous_policy']}",
            f"gen_mode-{self.config['policy_generator_mode']}",
            f"Pver-{self.config['prompt_ver']}",
            f"fewshot_num-{self.config['policy_generator_fewshots_num']}",
            f"cot-{self.config['enabled_cot']}",
            f"transfer-{self.enabled_transfer}",
            f"teacher-{self.enabled_teacher}"
        ])


def sanitize_log_data(d):
    """Sanitize log data by converting None values to empty strings and ensuring all values are of basic types.

    Args:
        d: Dictionary containing log data

    Returns:
        dict: Sanitized dictionary with safe values
    """
    new_d = {}
    for k, v in d.items():
        if isinstance(v, (str, int, list)):
            new_d[k] = v
        elif v is None:
            new_d[k] = ""
        else:
            # Convert other types to string
            new_d[k] = str(v)
    return new_d
