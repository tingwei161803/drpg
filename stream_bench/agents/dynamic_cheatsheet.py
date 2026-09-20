import re

from stream_bench.agents.base import Agent
from stream_bench.agents.policy import sanitize_log_data
from stream_bench.agents.utils import RAG


class DynamicCheatsheetAgent(Agent):
    """Dynamic Cheatsheet (DC): a controlled ablation of DRPG WITHOUT environment feedback.

    Like DRPG (``PolicyAgent``), DC keeps a memory of past cases and synthesizes a
    reusable "policy" (the *cheatsheet*) that is injected into the answering prompt.
    The only difference is that DC never uses environment feedback (correctness labels):

      1. Stored cases contain only ``(question, self-answer)`` -- no correctness suffix.
      2. Retrieval uses plain similarity (no correct/incorrect separation).
      3. The cheatsheet generator prompt makes no reference to feedback/correctness.

    This isolates the contribution of environment feedback in DRPG. The implementation
    mirrors ``PolicyAgent`` closely so that DC and DRPG differ by exactly one variable.

    This is the retrieval-based variant only (parallel to DRPG's RAG structure); the
    cumulative variant of the original Dynamic Cheatsheet paper is intentionally omitted.
    """

    LOG_KEYS_FOR_PROMPT = [
        "input_pred",
        "output_pred",
        "logprobs_pred",
    ]

    def __init__(self, config: dict) -> None:
        # Feature flags (mirrors the PolicyAgent subset; feedback-related knobs removed:
        # no `mode`, no `policy_generator_mode`, no EWMA). `use_correct_fewshots` is
        # renamed to the feedback-neutral `use_fewshots` (show retrieved examples or not).
        self.seed = config["seed"]
        self.enabled_cot = config.get("enabled_cot", False)
        self.use_fewshots = config.get("use_fewshots", True)
        self.use_previous_policy = config["use_previous_policy"]
        self.prompt_ver = config["prompt_ver"]
        self.policy_generator_fewshots_num = config["policy_generator_fewshots_num"]
        self.enabled_transfer = config.get("enabled_transfer", False)
        self.enabled_teacher = config.get("enabled_teacher", False)

        # Initialize base agent (after all attributes are set)
        super().__init__(config)

        self.llm_config = config["llm"]
        self.policy_llm_config = {
            "max_tokens": config["policy_llm"]["max_tokens"],
            "model_name": config["policy_llm"]["model_name"],
            "series": config["policy_llm"]["series"],
            "temperature": config["policy_llm"]["temperature"],
        }

        # Initialize RAG
        if config["rag"]["rag_filename"] is None:
            config["rag"]["rag_filename"] = self.log_path + ".db"
        self.rag = RAG(config["rag"])

        # State
        self.update_cnt = 0
        self.current_policy = None
        self.config = config

    def switch_models(self) -> None:
        """Switch from primary models to secondary models (for transfer experiments)."""
        if self.llm2 is not None:
            print("Switching from primary LLM to secondary LLM")
            self.llm = self.llm2
            self.llm_config = self.llm2_config
        if self.policy_llm2 is not None:
            print("Switching from primary policy LLM to secondary policy LLM")
            self.policy_llm = self.policy_llm2
            self.policy_llm_config = self.policy_llm2_config

    def generate_cheatsheet(self, cases: list[str], **kwargs) -> tuple[str, dict, str, str]:
        """Synthesize a cheatsheet (policy) from undifferentiated retrieved cases.

        Unlike DRPG's ``generate_rag_policy``, cases are NOT split into correct/incorrect;
        they are fed as a single neutral ``{previous_cases}`` block.

        Returns:
            tuple[str, dict, str, str]: policy, token info, policy prompt, raw policy output
        """
        template_name = (
            f"dc_policy_{'with' if self.use_previous_policy else 'no'}_prev_policy_"
            f"rule_ver{'1' if self.prompt_ver == 'original' else 'LEAP'}"
        )
        template = kwargs.get(template_name)

        format_params = {"previous_cases": "\n\n".join(cases)}
        if self.use_previous_policy:
            format_params["previous_policy"] = (
                self.current_policy if self.current_policy else "No previous policy"
            )

        prompt = template.format(**format_params)

        policy_text, policy_info = self.policy_llm(
            prompt=prompt,
            max_tokens=self.policy_llm_config["max_tokens"],
            temperature=self.policy_llm_config["temperature"],
        )

        policy_match = re.search(r"POLICY:\s*(.*)", policy_text, re.DOTALL)
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

        # Retrieve relevant past cases (no correctness separation)
        if self.rag.insert_acc > 0:
            cases = self.rag.retrieve(query=question, top_k=self.rag.top_k)
        else:
            cases = []

        # Select answering-side templates (CoT vs not). These already exist in the
        # benchmark and are feedback-neutral, so they are reused unchanged.
        if self.enabled_cot:
            template_wo_rag = prompt_cot
            template_w_rag_with_policy = kwargs.get("fewshotcot_template_with_policy")
            template_wo_rag_with_policy = kwargs.get("prompt_cot_with_policy")
        else:
            template_wo_rag = prompt_zeroshot
            template_w_rag_with_policy = kwargs.get("fewshot_template_with_policy")
            template_wo_rag_with_policy = kwargs.get("prompt_zeroshot_with_policy")

        # How many retrieved cases feed the cheatsheet generator
        if self.policy_generator_fewshots_num == "half_topk":
            fewshots_limit = self.rag.top_k
        elif self.policy_generator_fewshots_num == "two":
            fewshots_limit = 2
        elif self.policy_generator_fewshots_num == "one":
            fewshots_limit = 1
        else:
            fewshots_limit = 1
        cases_for_policy = cases[:fewshots_limit]

        policy_info = None
        policy_prompt = ""
        policy_raw_output = ""

        if len(cases) > 0:
            self.current_policy, policy_info, policy_prompt, policy_raw_output = (
                self.generate_cheatsheet(cases=cases_for_policy, **kwargs)
            )
            if self.current_policy is not None and self.use_fewshots:
                fewshot_text = "\n\n\n".join(cases).replace("\\", "\\\\")
                try:
                    prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=template_w_rag_with_policy)
                    prompt = re.sub(pattern=r"\{policy\}", repl=self.current_policy, string=prompt)
                except Exception as e:
                    error_msg = f"Error ```{e}``` caused by these cases. Logged to jsonl."
                    print(error_msg)
                    cases.append(error_msg)  # For analyzing errors afterwards
                    prompt = template_wo_rag
            elif self.current_policy is not None and not self.use_fewshots:
                try:
                    prompt = re.sub(pattern=r"\{policy\}", repl=self.current_policy, string=template_wo_rag_with_policy)
                except Exception as e:
                    error_msg = f"Error ```{e}``` caused by these cases. Logged to jsonl."
                    print(error_msg)
                    cases.append(error_msg)
                    prompt = template_wo_rag
            else:
                prompt = template_wo_rag
        else:
            if self.current_policy is not None:
                prompt = re.sub(pattern=r"\{policy\}", repl=self.current_policy, string=template_wo_rag_with_policy)
            else:
                prompt = template_wo_rag

        # Inference
        pred_text_raw, pred_info = self.llm(
            prompt=prompt,
            max_tokens=self.llm_config["max_tokens"],
            temperature=self.llm_config["temperature"]
        )

        # Filter pred_text (same heuristics as DRPG)
        pred_text = pred_text_raw
        if "</think>" in pred_text_raw:
            think_index = pred_text_raw.find("</think>")
            pred_text = pred_text_raw[think_index + len("</think>"):].strip()
        elif "ANSWER:" in pred_text_raw:
            answer_index = pred_text_raw.find("ANSWER:")
            pred_text = pred_text_raw[answer_index + len("ANSWER:"):].strip()

        # Logging
        self.update_log_info(log_data=sanitize_log_data({
            "num_shots": str(len(cases)),
            "retrieved_shots": cases,
            "input_pred": prompt,
            "output_pred": pred_text,
            "output_pred_raw": pred_text_raw,
            "logprobs_pred": pred_info["logprobs"],
            "num_inference_call": 1,
            "num_success_call": 1 if (pred_text[:5] != "error") else 0,
            "num_input_tokens": pred_info["num_input_tokens"] + (policy_info["num_input_tokens"] if policy_info else 0),
            "num_output_tokens": pred_info["num_output_tokens"] + (policy_info["num_output_tokens"] if policy_info else 0),
            "num_thinking_tokens": pred_info.get("num_thinking_tokens", 0) + (policy_info.get("num_thinking_tokens", 0) if policy_info else 0),
            "current_policy": str(self.current_policy),
            "policy_prompt": policy_prompt,
            "policy_raw_output": policy_raw_output,
        }))
        return pred_text

    def update(self, has_feedback: bool, **feedbacks) -> bool:
        """Store the (question, self-answer) pair in memory. No correctness is used."""
        question = feedbacks["question"]
        answer = feedbacks["self_output"]
        # Feedback-neutral memory chunk: question + the agent's own answer only.
        chunk = f"{question}\nYour answer: {answer}"

        if (self.bench.DATASET_PATH == "hotpot_qa") and (self.bench.DATASET_NAME == "distractor"):
            index = question.index("Question:")
            question = question[index:].strip()
            if self.update_cnt == 0:
                print("Embed question using only the question itself.")

        self.rag.insert(key=question, value=chunk)
        self.update_cnt += 1
        return True

    def get_name(self) -> str:
        return "__".join([
            self.config["agent_name"],
            self.llm_config["series"],
            self.llm_config["model_name"],
            self.config["rag"]["embedding_model"].split('/')[-1],  # remove '/' to avoid incorrect path
            self.config["rag"]["order"],
            str(self.config["rag"]["top_k"]),
            f"seed-{self.config['seed']}",
            f"use_fewshots-{self.use_fewshots}",
            f"use_prev_policy-{self.config['use_previous_policy']}",
            f"Pver-{self.config['prompt_ver']}",
            f"fewshot_num-{self.config['policy_generator_fewshots_num']}",
            f"cot-{self.config['enabled_cot']}",
            f"transfer-{self.enabled_transfer}",
            f"teacher-{self.enabled_teacher}",
        ])
