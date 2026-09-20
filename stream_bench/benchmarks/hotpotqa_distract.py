import copy
import json
import random
import re
import string
import textwrap

from datasets import Dataset
from stream_bench.benchmarks.base import Bench
from stream_bench.benchmarks.utils import extract_json_string, strip_all_lines


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction: str, truth: str) -> int:
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction: str, truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)

class HotpotQADistract(Bench):
    DATASET_PATH = "appier-ai-research/StreamBench"
    DATASET_NAME = "hotpotqa_distract"
    TEST_SIZE = 1500
    NUM_SHOTS = 4
    instruction_template = """You are optimizing a QA agent's policy to reduce errors."""
    ref_previous_policy = """
        Given the following previous policy:
        '''
        {previous_policy}
        '''"""
    ref_new_wrong_case = """
        Here is a new error case that has occurred:
        '''
        {new_wrong_case}
        '''"""
    ref_rag_wrong_cases = """
        Here are several error cases that have occurred:
        '''
        {previous_wrong_cases}
        '''"""
    ref_rag_correct_cases = """
        Here are several correct cases that have occurred:
        '''
        {previous_correct_cases}
        '''"""
    rules_ver1 = """
        Please analyze all the cases and revise the policy accordingly.
        Focus on question answering correctness and how to pass the required tests.
        Remove redundant or obsolete points. If certain previous policies are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise, actionable bullet points relevant to common QA mistakes.

        Example output format:
        POLICY:
        - [first point]
        - [second point]"""
    rules_verLEAP = """
        Conduct a thorough analysis of the generated answer and feedback of environment.
        Identify any discrepancies, misunderstandings, or errors present in the generated response.
        Based on this analysis, derive general policy that can be applied to improve future responses, focusing not just on the specific example but on the broader lessons.
        Focus on capturing the essence of the feedback while eliminating redundancies. Ensure that each point is clear, concise, and directly derived from the introspection results.

        Example output format:
        REASONING: <discuss why the generated answer is correct or wrong>
        POLICY: <create a bulletedlist (maximum 8) of unique, clear, and concise policy>"""
    # --- Dynamic Cheatsheet (DC): feedback-free building blocks (no correct/incorrect, no environment feedback) ---
    ref_rag_neutral_cases = """
        Here are several previous cases:
        '''
        {previous_cases}
        '''"""
    rules_ver1_dc = """
        Please analyze all the previous cases and revise the policy accordingly.
        Distill reusable strategies and common pitfalls for answering questions from these cases.
        Remove redundant or obsolete points. If certain previous policies are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise, actionable bullet points relevant to common QA situations.

        Example output format:
        POLICY:
        - [first point]
        - [second point]"""
    rules_verLEAP_dc = """
        Conduct a thorough analysis of the generated answers in the previous cases.
        Identify recurring patterns, useful strategies, and common pitfalls in solving these problems.
        Based on this analysis, derive general policy that can be applied to improve future responses, focusing not just on the specific examples but on the broader lessons.
        Focus on capturing reusable strategies while eliminating redundancies. Ensure that each point is clear, concise, and directly derived from the analysis.

        Example output format:
        REASONING: <discuss the strategies and patterns observed in the cases>
        POLICY: <create a bulletedlist (maximum 8) of unique, clear, and concise policy>"""

    def __init__(
        self,
        split: str = "validation",
        seed: int = 0,
        feedback: str = "correctness",
        setting: str = "gold_only",  # gold_only, distractor
        **kwargs
    ) -> None:
        assert setting in {"gold_only", "distractor"}
        self.setting = setting
        super().__init__({})
        self.split = split
        self.seed = seed
        self.feedback = feedback
        self.ems = list()
        self.f1s = list()
        self.fewshot_text = self.get_fewshot_text()

    def get_dataset(self) -> Dataset:
        # select the first TEST_SIZE examples from the dataset
        return self.dataset[self.split].select(range(self.TEST_SIZE)).shuffle(seed=self.seed)

    def get_input(self, row: dict) -> dict:
        """Builds the prompt for the LM to generate from."""
        row_input = copy.deepcopy(row)
        context = self.get_context(row)
        question = row["question"].strip()
        row_input["question"] = question
        row_input["prompt_zeroshot"] = self.get_zeroshot_prompt(context=context, question=question)
        row_input["prompt_fewshot"] = self.get_fewshot_prompt(context=context, question=question)
        row_input["prompt_cot"] = self.get_cot_prompt(context=context, question=question)
        row_input["fewshot_template"] = self.get_fewshot_template(context=context, question=question)
        row_input["fewshotcot_template"] = self.get_fewshotcot_template(context=context, question=question)
        row_input["feedback_template"] = self.get_feedback_template(context=context, question=question)
        row_input["refine_template"] = self.get_refine_template(context=context, question=question)
        row_input["critic_feedback_template"] = self.get_critic_feedback_template(context=context, question=question)
        row_input["critic_refine_template"] = self.get_critic_refine_template(context=context, question=question)
        row_input["fewshot_template_with_policy"] = self.get_fewshot_template_with_policy(context=context, question=question)
        row_input["prompt_zeroshot_with_policy"] = self.get_zeroshot_prompt_with_policy(context=context, question=question)
        row_input["prompt_cot_with_policy"] = self.get_cot_prompt_with_policy(context=context, question=question)
        row_input["fewshotcot_template_with_policy"] = self.get_fewshotcot_template_with_policy(context=context, question=question)

        # Add policy generator functions
        row_input["posthoc_policy_generator"] = self.posthoc_policy_generator()
        row_input["posthoc_policy_generator_with_optimization"] = self.posthoc_policy_generator_with_optimization()
        row_input["posthoc_policy_generator_with_correct_cases"] = self.posthoc_policy_generator_with_correct_cases()
        row_input["posthoc_policy_generator_with_optimization_and_correct_cases"] = self.posthoc_policy_generator_with_optimization_and_correct_cases()
        row_input["rag_policy_generator"] = self.rag_policy_generator()
        row_input["rag_policy_generator_with_optimization"] = self.rag_policy_generator_with_optimization()
        row_input["rag_policy_generator_with_correct_cases"] = self.rag_policy_generator_with_correct_cases()
        row_input["rag_policy_generator_with_optimization_and_correct_cases"] = self.rag_policy_generator_with_optimization_and_correct_cases()

        # Add policy rule functions for ver1 and verLEAP
        row_input["posthoc_policy_with_prev_policy_with_wrong_cases_rule_ver1"] = self.posthoc_policy_with_prev_policy_with_wrong_cases_rule_ver1()
        row_input["posthoc_policy_with_prev_policy_with_wrong_cases_rule_verLEAP"] = self.posthoc_policy_with_prev_policy_with_wrong_cases_rule_verLEAP()
        row_input["posthoc_policy_with_prev_policy_with_correct_cases_rule_ver1"] = self.posthoc_policy_with_prev_policy_with_correct_cases_rule_ver1()
        row_input["posthoc_policy_with_prev_policy_with_correct_cases_rule_verLEAP"] = self.posthoc_policy_with_prev_policy_with_correct_cases_rule_verLEAP()
        row_input["posthoc_policy_with_prev_policy_with_both_cases_rule_ver1"] = self.posthoc_policy_with_prev_policy_with_both_cases_rule_ver1()
        row_input["posthoc_policy_with_prev_policy_with_both_cases_rule_verLEAP"] = self.posthoc_policy_with_prev_policy_with_both_cases_rule_verLEAP()
        row_input["posthoc_policy_no_prev_policy_with_wrong_cases_rule_ver1"] = self.posthoc_policy_no_prev_policy_with_wrong_cases_rule_ver1()
        row_input["posthoc_policy_no_prev_policy_with_wrong_cases_rule_verLEAP"] = self.posthoc_policy_no_prev_policy_with_wrong_cases_rule_verLEAP()
        row_input["posthoc_policy_no_prev_policy_with_correct_cases_rule_ver1"] = self.posthoc_policy_no_prev_policy_with_correct_cases_rule_ver1()
        row_input["posthoc_policy_no_prev_policy_with_correct_cases_rule_verLEAP"] = self.posthoc_policy_no_prev_policy_with_correct_cases_rule_verLEAP()
        row_input["posthoc_policy_no_prev_policy_with_both_cases_rule_ver1"] = self.posthoc_policy_no_prev_policy_with_both_cases_rule_ver1()
        row_input["posthoc_policy_no_prev_policy_with_both_cases_rule_verLEAP"] = self.posthoc_policy_no_prev_policy_with_both_cases_rule_verLEAP()
        row_input["rag_policy_with_prev_policy_with_wrong_cases_rule_ver1"] = self.rag_policy_with_prev_policy_with_wrong_cases_rule_ver1()
        row_input["rag_policy_with_prev_policy_with_wrong_cases_rule_verLEAP"] = self.rag_policy_with_prev_policy_with_wrong_cases_rule_verLEAP()
        row_input["rag_policy_with_prev_policy_with_correct_cases_rule_ver1"] = self.rag_policy_with_prev_policy_with_correct_cases_rule_ver1()
        row_input["rag_policy_with_prev_policy_with_correct_cases_rule_verLEAP"] = self.rag_policy_with_prev_policy_with_correct_cases_rule_verLEAP()
        row_input["rag_policy_with_prev_policy_with_both_cases_rule_ver1"] = self.rag_policy_with_prev_policy_with_both_cases_rule_ver1()
        row_input["rag_policy_with_prev_policy_with_both_cases_rule_verLEAP"] = self.rag_policy_with_prev_policy_with_both_cases_rule_verLEAP()
        row_input["rag_policy_no_prev_policy_with_wrong_cases_rule_ver1"] = self.rag_policy_no_prev_policy_with_wrong_cases_rule_ver1()
        row_input["rag_policy_no_prev_policy_with_wrong_cases_rule_verLEAP"] = self.rag_policy_no_prev_policy_with_wrong_cases_rule_verLEAP()
        row_input["rag_policy_no_prev_policy_with_correct_cases_rule_ver1"] = self.rag_policy_no_prev_policy_with_correct_cases_rule_ver1()
        row_input["rag_policy_no_prev_policy_with_correct_cases_rule_verLEAP"] = self.rag_policy_no_prev_policy_with_correct_cases_rule_verLEAP()
        row_input["rag_policy_no_prev_policy_with_both_cases_rule_ver1"] = self.rag_policy_no_prev_policy_with_both_cases_rule_ver1()
        row_input["rag_policy_no_prev_policy_with_both_cases_rule_verLEAP"] = self.rag_policy_no_prev_policy_with_both_cases_rule_verLEAP()
        # Dynamic Cheatsheet (DC) policy generators
        row_input["dc_policy_no_prev_policy_rule_ver1"] = self.dc_policy_no_prev_policy_rule_ver1()
        row_input["dc_policy_no_prev_policy_rule_verLEAP"] = self.dc_policy_no_prev_policy_rule_verLEAP()
        row_input["dc_policy_with_prev_policy_rule_ver1"] = self.dc_policy_with_prev_policy_rule_ver1()
        row_input["dc_policy_with_prev_policy_rule_verLEAP"] = self.dc_policy_with_prev_policy_rule_verLEAP()

        return row_input

    def get_output(self, row: dict) -> str:
        return row["answer"].strip()

    def get_context(self, row: dict) -> str:
        """Get the context paragraphs for the given row.

        Format:
        Title: <title>
        Paragraph: <paragraph>

        Title: <title>
        Paragraph: <paragraph>
        ...
        """
        context = row["context"]
        assert len(context["title"]) == len(context["sentences"])
        if self.setting == "gold_only":
            titles = set(row["supporting_facts"]["title"])
        elif self.setting == "distractor":
            titles = set(context["title"])
        chunks = list()
        for title, sentences in zip(context["title"], context["sentences"]):
            if title in titles:
                title_text = f"Title: {title.strip()}"
                paragraph = f"Paragraph: {''.join(sentences).strip()}"
                chunk = '\n'.join([title_text, paragraph])
                chunks.append(chunk)
        return "\n\n".join(chunks)

    @staticmethod
    def get_zeroshot_prompt(context: str, question: str) -> str:
        """Get the zero-shot prompt for the given context and question"""
        prompt = textwrap.dedent(f"""\
        You are doing a question-answering task. You are given the following context, which might help you answer the question:

        Context (enclosed in triple backticks):
        ```
        {context}
        ```

        Question: {question}

        Note that you only need to answer with a short text span without explanation. Now, provide your answer in the following JSON format:
        {{"answer": "<your answer text span>"}}""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_cot_prompt(context: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are doing a question-answering task. You are given the following context, which might help you answer the question:

        Context (enclosed in triple backticks):
        ```
        {context}
        ```

        Question: {question}

        Let's take a deep breath and work on this problem step-by-step to derive the answer.
        Now, provide your answer in the following format:
        REASONING: <your rationale>
        ANSWER: <your answer text span, should only be a short text span without explanation.>""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template(context: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are doing a question-answering task. Here are some example cases:

        '''
        {{fewshot_text}}
        '''

        Now you are given the following context, which might help you answer the question:

        Context:
        ```
        {context}
        ```
        Question: {question}

        Note that you only need to answer with a short text span without explanation. Now, provide your answer in the following JSON format:
        {{"answer": "<your answer text span>"}}""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshotcot_template(context: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are doing a question-answering task. Here are some example cases:

        '''
        {{fewshot_text}}
        '''

        Now you are given the following context, which might help you answer the question:

        Context:
        ```
        {context}
        ```
        Question: {question}

        Now, take a deep breath and work on this problem step-by-step to derive the answer.
        Provide your output in the following format:
        REASONING: <your rationale>
        ANSWER: <your answer text span, should only be a short text span without explanation.>""")
        return strip_all_lines(prompt)


    def get_fewshot_prompt(self, context: str, question: str) -> str:
        fewshot_template = self.get_fewshot_template(context, question)
        return re.sub(r"\{fewshot_text\}", self.fewshot_text, fewshot_template)

    @staticmethod
    def get_question_text(context: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        Context:
        ```
        {context}
        ```
        Question: {question}""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_shot_template() -> str:
        prompt = textwrap.dedent(f"""\
        {{question}}
        {{answer}}""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_memprompt_template() -> str:
        prompt = textwrap.dedent(f"""\
        {{question}}
        Your answer: {{answer}}
        User Feedback: {{correctness}}""")
        return strip_all_lines(prompt)

    def get_fewshot_text(self) -> str:
        shot_rows = self.dataset["train"]
        shots = list()
        for i in range(self.NUM_SHOTS):
            row = shot_rows[i]
            shot = self.get_shot_template().format(
                question=self.get_question_text(context=self.get_context(row), question=row["question"].strip()),
                answer=self.get_label_text(row["answer"].strip())
            )
            shots.append(shot)
        print(f"{len(shots)} shots generated.")
        return "\n\n\n".join(shots).replace("\\", "\\\\")

    @staticmethod
    def get_feedback_template(context: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are doing a question-answering task. You are given the following context, which might help you answer the question:

        Context (enclosed in triple backticks):
        ```
        {context}
        ```

        Question: {question}

        Your answer: {{y_hat}}
        First, determine whether you need to refine your answer in terms of its correctness.
        If you consider that your answer is correct, output 'NO NEED TO REFINE' in uppercase.
        Otherwise, provide a suggestion to correct the answer. Your answer should be a short text span without explanation.""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_refine_template(context: str, question: str) -> str:
        """Note: The format of answer-feedback trajectory should be as follows
        Answer 0: <answer_0>
        Feedback 0: <feedback_0>
        Answer 1: <answer_1>
        Feedback 1: <feedback_1>
        ...
        Answer k: <answer_k>
        Feedback k: <feedback_k>
        """
        prompt = textwrap.dedent(f"""\
        You are doing a question-answering task. You are given the following context, which might help you answer the question:

        Context (enclosed in triple backticks):
        ```
        {context}
        ```

        Question: {question}

        -- Your previous answer-feedback trajectory:
        {{trajectory}}

        According to the latest feedback, provide your new answer.
        Your answer should be a short text span without explanation.""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_critic_feedback_template(context: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are an expert evaluator reviewing the correctness of an answer provided for a question-answering task. Consider the following context to help answer the question:

        Context (enclosed in triple backticks):
        ```
        {context}
        ```

        Question: {question}

        Provided answer: {{y_hat}}

        Review the provided answer considering previous similar cases listed below:
        {{previous_cases}}

        Determine clearly:
        1. Whether the provided answer should be refined or not.
        2. Briefly explain the reason for your decision based on the provided context, question, and previous cases.
        3. If refinement is needed, provide a suggestion to correct the answer as a short text span without explanation.

        If no refinement is needed, output 'NO NEED TO REFINE' in uppercase.""")
        return strip_all_lines(prompt)


    @staticmethod
    def get_critic_refine_template(context: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing a question-answering task and need to refine your answer based on expert feedback. Consider the following context to help answer the question:

        Context (enclosed in triple backticks):
        ```
        {context}
        ```

        Question: {question}

        Your previous answer:
        {{y_hat}}

        Feedback from expert evaluator:
        {{critic_rationale}}

        According to the feedback, provide your new refined answer.
        Your answer should be a short text span without explanation.""")
        return strip_all_lines(prompt)


    def get_label_text(self, label: str | dict) -> str:
        if isinstance(label, dict) and "answer" in label:
            return json.dumps(label)
        elif isinstance(label, str):
            return json.dumps({"answer": label})
        raise ValueError(f"Invalid label type: {type(label)}")

    def postprocess_generation(self, res: str, idx: int = -1) -> str:
        # Parse the answer out of Answer: '<answer>'
        json_str = extract_json_string(res)
        try:
            obj = json.loads(json_str)
            if "answer" not in obj:
                print(f"Failed to find 'answer' key in the response: {json_str}")
                answer = json_str
            else:
                answer = obj["answer"]
        except json.JSONDecodeError:
            print(f"Failed to parse answer from the response: {res}")
            answer = res
        return answer

    def process_results(
        self,
        prediction: str,
        label: str,
        return_details: bool = True,
        **kwargs
    ) -> dict | bool:
        em = compute_exact_match(prediction, label)
        f1 = compute_f1(prediction, label)
        self.ems.append(em)
        self.f1s.append(f1)
        if return_details:
            return {
                "correct": em,
                "rolling_em": sum(self.ems) / len(self.ems),
                "rolling_f1": sum(self.f1s) / len(self.f1s),
            }
        return em

    def get_metrics(self) -> dict:
        return {
            "em": sum(self.ems) / len(self.ems),
            "f1": sum(self.f1s) / len(self.f1s),
        }

    def give_feedback(self, model_output: str, row: dict, res: dict) -> tuple[bool, dict]:
        answer = self.postprocess_generation(model_output)
        has_feedback = True
        feedbacks = {
            "question": self.get_question_text(context=self.get_context(row), question=row["question"].strip()),
            "self_output": self.get_label_text(answer),
            "is_correct": res["correct"],
            "ground_truth": self.get_label_text(row["answer"].strip()),
            "shot_template": self.get_shot_template(),
            "memprompt_template": self.get_memprompt_template()
        }
        return has_feedback, feedbacks

    @staticmethod
    def get_fewshot_template_with_policy(context: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are doing a question-answering task. Here are some example cases:

        '''
        {{fewshot_text}}
        '''

        Please pay special attention to the following points, which are derived from previous cases.
        '''
        {{policy}}
        '''

        Now you are given the following context, which might help you answer the question:

        Context:
        ```
        {context}
        ```
        Question: {question}

        Note that you only need to answer with a short text span without explanation. Now, provide your answer in the following JSON format:
        {{"answer": "<your answer text span>"}}""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_zeroshot_prompt_with_policy(context: str, question: str) -> str:
        """Get the zero-shot prompt for the given context and question"""
        prompt = textwrap.dedent(f"""\
        You are doing a question-answering task.

        Please pay special attention to the following points, which are derived from previous cases.
        '''
        {{policy}}
        '''

        Now you are given the following context, which might help you answer the question:

        Context (enclosed in triple backticks):
        ```
        {context}
        ```

        Question: {question}

        Note that you only need to answer with a short text span without explanation. Now, provide your answer in the following JSON format:
        {{"answer": "<your answer text span>"}}""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshotcot_template_with_policy(context: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are doing a question-answering task. Here are some example cases:

        '''
        {{fewshot_text}}
        '''

        Please pay special attention to the following points, which are derived from previous cases.
        '''
        {{policy}}
        '''

        Now you are given the following context, which might help you answer the question:

        Context:
        ```
        {context}
        ```
        Question: {question}

        Now, take a deep breath and work on this problem step-by-step to derive the answer.
        Provide your output in the following format:
        REASONING: <your rationale>
        ANSWER: <your answer text span, should only be a short text span without explanation.>""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_cot_prompt_with_policy(context: str, question: str) -> str:
        """Get the zero-shot prompt for the given context and question"""
        prompt = textwrap.dedent(f"""\
        You are doing a question-answering task.

        Please pay special attention to the following points, which are derived from previous cases.
        '''
        {{policy}}
        '''

        Now you are given the following context, which might help you answer the question:

        Context (enclosed in triple backticks):
        ```
        {context}
        ```

        Question: {question}

        Now, take a deep breath and work on this problem step-by-step to derive the answer.
        Provide your output in the following format:
        REASONING: <your rationale>
        ANSWER: <your answer text span, should only be a short text span without explanation.>""")
        return strip_all_lines(prompt)

    @staticmethod
    def posthoc_policy_generator() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing a QA agent's policy to reduce errors.

        Given the following previous policy:
        {previous_policy}

        Here is a new error case that has occurred:
        {new_wrong_case}

        Please analyze the new error case and revise the policy accordingly.
        Output the new policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        **Limit the revised policy to at most 5 concise bullet points. Only keep the most critical and actionable items.**

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def posthoc_policy_generator_with_optimization() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing a QA agent's policy to reduce errors.

        Given the following previous policy:
        {previous_policy}

        Here is a new error case that has occurred:
        {new_wrong_case}

        The effectiveness of the previous policy is measured by the following confidence score (Exponentially Weighted Moving Average accuracy):
        {confidence_score}

        - If the confidence score is high, only minor policy refinements may be needed.
        - If the confidence score is low, consider making more significant changes to the policy.

        Please analyze the new error case and revise the policy accordingly.
        Output the new policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        **Limit the revised policy to at most 5 concise bullet points. Only keep the most critical and actionable items.**

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def posthoc_policy_generator_with_correct_cases() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing a QA agent's policy to reduce errors.

        Given the following previous policy:
        {previous_policy}

        Here is a new error case that has occurred:
        {new_wrong_case}

        Here are some previous correct cases for reference:
        {previous_correct_case}

        Please analyze the new error case and revise the policy accordingly.
        Output the new policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        **Limit the revised policy to at most 5 concise bullet points. Only keep the most critical and actionable items.**

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def posthoc_policy_generator_with_optimization_and_correct_cases() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing a QA agent's policy to reduce errors.

        Given the following previous policy:
        {previous_policy}

        Here is a new error case that has occurred:
        {new_wrong_case}

        Here are some previous correct cases for reference:
        {previous_correct_case}

        The effectiveness of the previous policy is measured by the following confidence score (Exponentially Weighted Moving Average accuracy):
        {confidence_score}

        - If the confidence score is high, only minor policy refinements may be needed.
        - If the confidence score is low, consider making more significant changes to the policy.

        Please analyze the new error case and revise the policy accordingly.
        Output the new policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        **Limit the revised policy to at most 5 concise bullet points. Only keep the most critical and actionable items.**

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def rag_policy_generator() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing a QA agent's policy to reduce errors.

        Given the following previous policy:
        {previous_policy}

        Here are several error cases that have occurred:
        {previous_wrong_cases}

        Please analyze all the new error cases and revise the policy accordingly.
        Output the new policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        **Limit the revised policy to at most 5 concise bullet points. Only keep the most critical and actionable items.**

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def rag_policy_generator_with_optimization() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing a QA agent's policy to reduce errors.

        Given the following previous policy:
        {previous_policy}

        Here are several error cases that have occurred:
        {previous_wrong_cases}

        The effectiveness of the previous policy is measured by the following confidence score (Exponentially Weighted Moving Average accuracy):
        {confidence_score}

        - If the confidence score is high, only minor policy refinements may be needed.
        - If the confidence score is low, consider making more significant changes to the policy.

        Please analyze all the new error cases and revise the policy accordingly.
        Output the new policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        **Limit the revised policy to at most 5 concise bullet points. Only keep the most critical and actionable items.**

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def rag_policy_generator_with_correct_cases() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing a QA agent's policy to reduce errors.

        Given the following previous policy:
        {previous_policy}

        Here are several error cases that have occurred:
        {previous_wrong_cases}

        Here are some previous correct cases for reference:
        {previous_correct_case}

        Please analyze all the new error cases and revise the policy accordingly.
        Output the new policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        **Limit the revised policy to at most 5 concise bullet points. Only keep the most critical and actionable items.**

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def rag_policy_generator_with_optimization_and_correct_cases() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing a QA agent's policy to reduce errors.

        Given the following previous policy:
        {previous_policy}

        Here are several error cases that have occurred:
        {previous_wrong_cases}

        Here are some previous correct cases for reference:
        {previous_correct_case}

        The effectiveness of the previous policy is measured by the following confidence score (Exponentially Weighted Moving Average accuracy):
        {confidence_score}

        - If the confidence score is high, only minor policy refinements may be needed.
        - If the confidence score is low, consider making more significant changes to the policy.

        Please analyze all the new error cases and revise the policy accordingly.
        Output the new policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        **Limit the revised policy to at most 5 concise bullet points. Only keep the most critical and actionable items.**

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    def posthoc_policy_with_prev_policy_with_wrong_cases_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_new_wrong_case + self.rules_ver1
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def posthoc_policy_with_prev_policy_with_wrong_cases_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_new_wrong_case + self.rules_verLEAP
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def posthoc_policy_with_prev_policy_with_correct_cases_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_rag_correct_cases + self.rules_ver1
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def posthoc_policy_with_prev_policy_with_correct_cases_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_rag_correct_cases + self.rules_verLEAP
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def posthoc_policy_with_prev_policy_with_both_cases_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_new_wrong_case + self.ref_rag_correct_cases + self.rules_ver1
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def posthoc_policy_with_prev_policy_with_both_cases_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_new_wrong_case + self.ref_rag_correct_cases + self.rules_verLEAP
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def posthoc_policy_no_prev_policy_with_wrong_cases_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_new_wrong_case + self.rules_ver1
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def posthoc_policy_no_prev_policy_with_wrong_cases_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_new_wrong_case + self.rules_verLEAP
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def posthoc_policy_no_prev_policy_with_correct_cases_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_rag_correct_cases + self.rules_ver1
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def posthoc_policy_no_prev_policy_with_correct_cases_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_rag_correct_cases + self.rules_verLEAP
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def posthoc_policy_no_prev_policy_with_both_cases_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_new_wrong_case + self.ref_rag_correct_cases + self.rules_ver1
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def posthoc_policy_no_prev_policy_with_both_cases_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_new_wrong_case + self.ref_rag_correct_cases + self.rules_verLEAP
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def rag_policy_with_prev_policy_with_wrong_cases_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_rag_wrong_cases + self.rules_ver1
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def rag_policy_with_prev_policy_with_wrong_cases_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_rag_wrong_cases + self.rules_verLEAP
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def rag_policy_with_prev_policy_with_correct_cases_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_rag_correct_cases + self.rules_ver1
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def rag_policy_with_prev_policy_with_correct_cases_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_rag_correct_cases + self.rules_verLEAP
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def rag_policy_with_prev_policy_with_both_cases_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_rag_wrong_cases + self.ref_rag_correct_cases + self.rules_ver1
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def rag_policy_with_prev_policy_with_both_cases_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_rag_wrong_cases + self.ref_rag_correct_cases + self.rules_verLEAP
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def rag_policy_no_prev_policy_with_wrong_cases_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_rag_wrong_cases + self.rules_ver1
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def rag_policy_no_prev_policy_with_wrong_cases_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_rag_wrong_cases + self.rules_verLEAP
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def rag_policy_no_prev_policy_with_correct_cases_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_rag_correct_cases + self.rules_ver1
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def rag_policy_no_prev_policy_with_correct_cases_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_rag_correct_cases + self.rules_verLEAP
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def rag_policy_no_prev_policy_with_both_cases_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_rag_wrong_cases + self.ref_rag_correct_cases + self.rules_ver1
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def rag_policy_no_prev_policy_with_both_cases_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_rag_wrong_cases + self.ref_rag_correct_cases + self.rules_verLEAP
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    # --- Dynamic Cheatsheet (DC): feedback-free policy generators (neutral cases, no correctness) ---
    def dc_policy_no_prev_policy_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_rag_neutral_cases + self.rules_ver1_dc
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def dc_policy_no_prev_policy_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_rag_neutral_cases + self.rules_verLEAP_dc
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def dc_policy_with_prev_policy_rule_ver1(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_rag_neutral_cases + self.rules_ver1_dc
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)

    def dc_policy_with_prev_policy_rule_verLEAP(self) -> str:
        prompt = self.instruction_template + self.ref_previous_policy + self.ref_rag_neutral_cases + self.rules_verLEAP_dc
        prompt = textwrap.dedent(prompt)
        return strip_all_lines(prompt)