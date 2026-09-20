import copy
import re
import textwrap
import warnings

warnings.filterwarnings("default")

from stream_bench.benchmarks.base import Bench
from stream_bench.benchmarks.ds1000_utils import execution, fewshots
from stream_bench.benchmarks.utils import strip_all_lines

four_space_tab = "    "

def fix_tab_indents(text: str) -> str:
    return text.replace("\t", four_space_tab)

def extract_code(text):
    pattern = r'```(?:python)?\s*(.*?)```?'
    code_blocks = re.findall(pattern, text, re.DOTALL)
    return code_blocks


class DS1000(Bench):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    DATASET_PATH = "appier-ai-research/StreamBench"
    DATASET_NAME = "ds_1000"
    FEWSHOTS = fewshots.rows
    instruction_template = """You are optimizing the policy for a Python programming agent to reduce errors in solving user requirements."""
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
        Focus on Python code correctness and how to pass the required tests.
        Remove redundant or obsolete points. If certain previous policies are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise, actionable bullet points relevant to common Python coding and testing mistakes.

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
        Distill reusable strategies and common pitfalls for solving Python programming requirements from these cases.
        Remove redundant or obsolete points. If certain previous policies are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise, actionable bullet points relevant to common Python coding situations.

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
        split: str = "test",
        seed: int = 0,
        feedback: str = "correctness",
        timeout=3.0,
        agent=None,
        mode='instruct',
        **kwargs
    ) -> None:
        super().__init__(config={})
        assert mode in ['instruct', 'write','replit-glaive','standard']
        # outsource this to yaml attribute
        self.split = split
        self.seed = seed
        self.feedback = feedback
        self.total = 0
        self.correct_stats = {lib: [] for lib in ['Pytorch', 'Tensorflow', 'Pandas', 'Numpy', 'Matplotlib', 'Sklearn', 'Scipy']}
        self.timeout = timeout
        self.agent_callback = None
        if hasattr(agent, 'retrieve_experience'):
            self.agent_callback = agent.retrieve_experience
        self.fewshot_text = self.get_fewshot_text()

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset[self.split].shuffle(seed=self.seed)

    # Next, I would like to refactor the following code to be like the class GeneralText2SQL
    @staticmethod
    def get_zeroshot_prompt(
        question: str,
        exec_context: str
    ) -> str:
        prompt = textwrap.dedent(f"""\
        Here is the user's requirements for solving a programming problem (enclosed in '''):
        '''
        {question}
        '''

        You need to provide your solution in python code to satisfy the user's requirements. Your code will be tested as follows (enclosed in '''):
        '''
        {exec_context}

        code = exec_context.replace("[insert]", <your_code>)
        a_test_case = generate_test_case()
        test_input, expected_result = a_test_case
        test_env = {{"test_input": test_input}}
        exec(code, test_env)
        assertEqual(test_env["result"], expected_result)
        '''

        Now, generate your code directly in the following format:
        ```python
        <your_code>
        ```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_cot_prompt(
        question: str,
        exec_context: str
    ) -> str:
        prompt = textwrap.dedent(f"""\
        Here is the user's requirements for solving a programming problem (enclosed in '''):
        '''
        {question}
        '''

        You need to provide your solution in python code to satisfy the user's requirements. Your code will be tested as follows (enclosed in '''):
        '''
        {exec_context}

        code = exec_context.replace("[insert]", <your_code>)
        a_test_case = generate_test_case()
        test_input, expected_result = a_test_case
        test_env = {{"test_input": test_input}}
        exec(code, test_env)
        assertEqual(test_env["result"], expected_result)
        '''

        Now, take a deep breath and work on this problem step-by-step to derive the correct python code.
        Provide your output in the following format:
        REASONING:  <your_rationale>
        ANSWER: ```python
        <your_code>
        ```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_shot_template() -> str:
        prompt = textwrap.dedent(f"""\
        The user's requirements:
        '''
        {{question}}
        '''
        Your solution in python code:
        ```python
        {{answer}}
        ```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_memprompt_template() -> str:
        prompt = textwrap.dedent(f"""\
        The user's requirements:
        '''
        {{question}}
        '''
        Your solution in python code:
        ```python
        {{answer}}
        ```
        User Feedback: {{correctness}}""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template(
        question: str,
        exec_context: str
    ) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing a python programming task to satisfy the user's requirements. Here are some examples:

        '''
        {{fewshot_text}}
        '''

        Now it's your turn.

        The user's requirements:
        '''
        {question}
        '''

        You need to provide your solution in python code to satisfy the user's requirements. Your code will be tested as follows (enclosed in '''):
        '''
        {exec_context}

        code = exec_context.replace("[insert]", <your_code>)
        a_test_case = generate_test_case()
        test_input, expected_result = a_test_case
        test_env = {{"test_input": test_input}}
        exec(code, test_env)
        assertEqual(test_env["result"], expected_result)
        '''

        Now, generate your code directly in the following format:
        ```python
        <your_code>
        ```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshotcot_template(
        question: str,
        exec_context: str
    ) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing a python programming task to satisfy the user's requirements. Here are some examples:

        '''
        {{fewshot_text}}
        '''

        Now it's your turn.

        The user's requirements:
        '''
        {question}
        '''

        You need to provide your solution in python code to satisfy the user's requirements. Your code will be tested as follows (enclosed in '''):
        '''
        {exec_context}

        code = exec_context.replace("[insert]", <your_code>)
        a_test_case = generate_test_case()
        test_input, expected_result = a_test_case
        test_env = {{"test_input": test_input}}
        exec(code, test_env)
        assertEqual(test_env["result"], expected_result)
        '''

        Now, take a deep breath and work on this problem step-by-step to derive the correct python code.
        Provide your output in the following format:
        REASONING:  <your_rationale>
        ANSWER: ```python
        <your_code>
        ```""")
        return strip_all_lines(prompt)

    def get_fewshot_prompt(
        self,
        question: str,
        exec_context: str
    ) -> str:
        fewshot_template = self.get_fewshot_template(question=question, exec_context=exec_context)
        return re.sub(pattern=r"\{fewshot_text\}", repl=self.fewshot_text, string=fewshot_template)

    def get_fewshot_text(self) -> str:
        assert len(self.FEWSHOTS) == 4
        shots = list()
        for row in self.FEWSHOTS:
            shot = self.get_shot_template().format(
                question=row["prompt"].strip(),
                answer=row["reference_code"]
            )
            shots.append(shot)
        return "\n\n\n".join(shots).replace("\\", "\\\\")

    def get_input(self, row: dict) -> dict:
        """Builds the prompt for the LM to generate from."""
        row_input = copy.deepcopy(row)
        exec_context = row["code_context"]
        row_input["question"] = row["prompt"].strip()
        row_input["label_text"] = row["reference_code"]
        row_input["prompt_zeroshot"] = self.get_zeroshot_prompt(question=row_input["question"], exec_context=exec_context)
        row_input["prompt_fewshot"] = self.get_fewshot_prompt(question=row_input["question"], exec_context=exec_context)
        row_input["prompt_cot"] = self.get_cot_prompt(question=row_input["question"], exec_context=exec_context)
        row_input["fewshot_template"] = self.get_fewshot_template(question=row_input["question"], exec_context=exec_context)
        row_input["fewshotcot_template"] = self.get_fewshotcot_template(question=row_input["question"], exec_context=exec_context)
        row_input["feedback_template"] = self.get_feedback_template(question=row_input["question"], exec_context=exec_context)
        row_input["refine_template"] = self.get_refine_template(question=row_input["question"], exec_context=exec_context)
        row_input["critic_feedback_template"] = self.get_critic_feedback_template(question=row_input["question"], exec_context=exec_context)
        row_input["critic_refine_template"] = self.get_critic_refine_template(question=row_input["question"], exec_context=exec_context)
        row_input["fewshot_template_with_policy"] = self.get_fewshot_template_with_policy(question=row_input["question"], exec_context=exec_context)
        row_input["prompt_zeroshot_with_policy"] = self.get_zeroshot_prompt_with_policy(question=row_input["question"], exec_context=exec_context)
        row_input["prompt_cot_with_policy"] = self.get_cot_prompt_with_policy(question=row_input["question"], exec_context=exec_context)
        row_input["fewshotcot_template_with_policy"] = self.get_fewshotcot_template_with_policy(question=row_input["question"], exec_context=exec_context)

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

    # STEP 2
    def postprocess_generation(self, generation: str, idx: int = -1) -> str:
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        try:
            match = re.search(
                pattern=r"```python(.*?)```",
                string=generation,
                flags=re.DOTALL
            ).group(1)
        except Exception as e:
            match = generation
        return match

    # STEP 3
    def get_output(self, doc):
        return {
            'label': doc['reference_code'],
            'problem_id' : int(doc['metadata']['problem_id']),
            'lib' : doc['metadata']['library'],
            'code_context': doc['code_context']
        }

    # STEP 4
    def process_results(self, generations, references, return_details=False, simulate_env=False,**kwargs):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_context = references['code_context']
        prediction = generations
        completion_id = references['problem_id']
        lib = references['lib']
        test_program = (
            "import matplotlib\nmatplotlib.use('agg')\n" + code_context + '\n'
            + f'code = {repr(prediction)}\n'
            + 'test_execution(code)\n'
            + ('test_string(code)\n'  if 'test_string(' in code_context  else '\n')
        )
        results = execution.check_correctness(
            test_program,
            timeout=self.timeout,
            completion_id=completion_id
        )
        correct = results['passed']

        if not simulate_env:
            self.correct_stats[lib].append(int(correct))
            self.n_correct += int(correct)
            self.predictions.append(generations)
            self.references.append(references)
            self.total += 1
        rolling_acc = self.n_correct / max(self.total, 1)

        if return_details:
            stats = {}
            for lib, correct_counts in self.correct_stats.items():
                stats[lib] = sum(correct_counts)/max(len(correct_counts), 1)
            return {
                'correct': correct,
                'n_correct': self.n_correct,
                'rolling_acc': rolling_acc,
                **stats,
                **results
            }
        return correct

    def give_feedback(self, model_output: str, row: dict, res: dict) -> tuple[bool, dict]:
        pred_str = self.postprocess_generation(model_output)
        has_feedback = True
        feedbacks = {
            "question": row["prompt"].strip(),
            "self_output": pred_str,
            "is_correct": res["correct"],
            "ground_truth": row["reference_code"],
            "shot_template": self.get_shot_template(),
            "memprompt_template": self.get_memprompt_template()
        }
        return has_feedback, feedbacks

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    def get_metrics(self):
        return {
            'pass@1': self.n_correct / self.total
        }

    def get_question_text(self, row):
        return row['raw_prompt']['prompt']

    def get_answer_text(self, row):
        return row["reference_code"]

    def get_choices_text(self, row):
        return ''

    @staticmethod
    def extract_after_exec_content(code_context: str) -> str:
        return re.findall(pattern=r'exec_context.*"""', string=code_context, flags=re.DOTALL)

    @staticmethod
    def get_feedback_template(question: str, exec_context: str) -> str:
        prompt = textwrap.dedent(f"""\
        Here is the user's requirements for solving a programming problem (enclosed in '''):
        '''
        {question}
        '''
        You need to provide your solution in python code to satisfy the user's requirements. Your code will be tested as follows (enclosed in '''):
        '''
        {exec_context}
        code = exec_context.replace("[insert]", <your_code>)
        a_test_case = generate_test_case()
        test_input, expected_result = a_test_case
        test_env = {{"test_input": test_input}}
        exec(code, test_env)
        assertEqual(test_env["result"], expected_result)
        '''
        Here is your previously generated code:
        ```python
        {{y_hat}}
        ```
        First, determine whether you need to refine your code in terms of its correctness.
        If you consider that your code is correct, output 'NO NEED TO REFINE' in uppercase.
        Otherwise, provide a suggestion to correct the code.""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_refine_template(question: str, exec_context: str) -> str:
        """Note: The format of answer-feedback trajectory should be as follows
        Answer 0: <code_0>
        Feedback 0: <feedback_0>
        Answer 1: <code_1>
        Feedback 1: <feedback_1>
        ...
        Answer k: <code_n>
        Feedback k: <feedback_n>
        """
        prompt = textwrap.dedent(f"""\
        Here is the user's requirements for solving a programming problem (enclosed in '''):
        '''
        {question}
        '''
        You need to provide your solution in python code to satisfy the user's requirements. Your code will be tested as follows (enclosed in '''):
        '''
        {exec_context}
        code = exec_context.replace("[insert]", <your_code>)
        a_test_case = generate_test_case()
        test_input, expected_result = a_test_case
        test_env = {{"test_input": test_input}}
        exec(code, test_env)
        assertEqual(test_env["result"], expected_result)
        '''
        Here is your previous answer-feedback trajectory:
        {{trajectory}}
        According to the latest feedback, provide your refined code.
        Provide your output in the following format: ```python\n<refined_code>\n```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_critic_feedback_template(question: str, exec_context: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are an expert programmer reviewing the correctness of Python code for the given programming task. Here are the user's requirements (enclosed in '''):
        '''
        {question}
        '''

        Your code must satisfy the following execution context and testing scenario (enclosed in '''):
        '''
        {exec_context}
        code = exec_context.replace("[insert]", <your_code>)
        a_test_case = generate_test_case()
        test_input, expected_result = a_test_case
        test_env = {{"test_input": test_input}}
        exec(code, test_env)
        assertEqual(test_env["result"], expected_result)
        '''

        Here is your previously generated code:
        ```python
        {{y_hat}}
        ```

        Review the provided code considering previous similar cases listed below:
        {{previous_cases}}

        Determine clearly:
        1. Whether your previously generated code should be refined or not.
        2. Briefly explain the reason for your decision based on the user's requirements, the provided execution context, and previous cases.
        3. If refinement is needed, provide a suggestion on how to correct the code.""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_critic_refine_template(question: str, exec_context: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are a Python programmer tasked with refining your solution based on expert feedback. Here are the user's original requirements (enclosed in '''):
        '''
        {question}
        '''

        Your code must satisfy the following execution context and testing scenario (enclosed in '''):
        '''
        {exec_context}
        code = exec_context.replace("[insert]", <your_code>)
        a_test_case = generate_test_case()
        test_input, expected_result = a_test_case
        test_env = {{"test_input": test_input}}
        exec(code, test_env)
        assertEqual(test_env["result"], expected_result)
        '''

        Here is your previous code:
        ```python
        {{y_hat}}
        ```

        Feedback from expert reviewer:
        {{critic_rationale}}

        According to the feedback, provide your refined Python code.
        Provide your output in the following format: ```python\n<refined_code>\n```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template_with_policy(
        question: str,
        exec_context: str
    ) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing a python programming task to satisfy the user's requirements. Here are some examples:

        {{fewshot_text}}

        Please pay special attention to the following points, which are derived from previous cases.
        {{policy}}

        Now it's your turn.

        The user's requirements (enclosed in '''):
        '''
        {question}
        '''

        You need to provide your solution in python code to satisfy the user's requirements. Your code will be tested as follows (enclosed in '''):
        '''
        {exec_context}

        code = exec_context.replace("[insert]", <your_code>)
        a_test_case = generate_test_case()
        test_input, expected_result = a_test_case
        test_env = {{"test_input": test_input}}
        exec(code, test_env)
        assertEqual(test_env["result"], expected_result)
        '''

        Now, generate your code directly in the following format:
        ```python
        <your_code>
        ```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_zeroshot_prompt_with_policy(
        question: str,
        exec_context: str
    ) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing a python programming task to satisfy the user's requirements.

        Please pay special attention to the following points, which are derived from previous cases.
        {{policy}}

        Here is the user's requirements for solving a programming problem (enclosed in '''):
        '''
        {question}
        '''

        You need to provide your solution in python code to satisfy the user's requirements. Your code will be tested as follows (enclosed in '''):
        '''
        {exec_context}

        code = exec_context.replace("[insert]", <your_code>)
        a_test_case = generate_test_case()
        test_input, expected_result = a_test_case
        test_env = {{"test_input": test_input}}
        exec(code, test_env)
        assertEqual(test_env["result"], expected_result)
        '''

        Now, generate your code directly in the following format:
        ```python
        <your_code>
        ```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshotcot_template_with_policy(
        question: str,
        exec_context: str
    ) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing a python programming task to satisfy the user's requirements. Here are some examples:

        {{fewshot_text}}

        Please pay special attention to the following points, which are derived from previous cases.
        {{policy}}

        Now it's your turn.

        The user's requirements (enclosed in '''):
        '''
        {question}
        '''

        You need to provide your solution in python code to satisfy the user's requirements. Your code will be tested as follows (enclosed in '''):
        '''
        {exec_context}

        code = exec_context.replace("[insert]", <your_code>)
        a_test_case = generate_test_case()
        test_input, expected_result = a_test_case
        test_env = {{"test_input": test_input}}
        exec(code, test_env)
        assertEqual(test_env["result"], expected_result)
        '''

        Now, take a deep breath and work on this problem step-by-step to derive the correct python code.
        Provide your output in the following format:
        REASONING:  <your_rationale>
        ANSWER: ```python
        <your_code>
        ```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_cot_prompt_with_policy(
        question: str,
        exec_context: str
    ) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing a python programming task to satisfy the user's requirements.

        Please pay special attention to the following points, which are derived from previous cases.
        {{policy}}

        Here is the user's requirements for solving a programming problem (enclosed in '''):
        '''
        {question}
        '''

        You need to provide your solution in python code to satisfy the user's requirements. Your code will be tested as follows (enclosed in '''):
        '''
        {exec_context}

        code = exec_context.replace("[insert]", <your_code>)
        a_test_case = generate_test_case()
        test_input, expected_result = a_test_case
        test_env = {{"test_input": test_input}}
        exec(code, test_env)
        assertEqual(test_env["result"], expected_result)
        '''

        Now, take a deep breath and work on this problem step-by-step to derive the correct python code.
        Provide your output in the following format:
        REASONING:  <your_rationale>
        ANSWER: ```python
        <your_code>
        ```""")
        return strip_all_lines(prompt)

    @staticmethod
    def posthoc_policy_generator() -> str:
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a Python programming agent to reduce errors in solving user requirements.

        Given the following previous policy:
        {previous_policy}

        Here is a new error case that has occurred:
        {new_wrong_case}

        Please analyze the new error case and revise the policy accordingly.
        Focus on Python code correctness and how to pass the required tests.
        Remove redundant or obsolete points. If certain previous policies are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise, actionable bullet points relevant to common Python coding and testing mistakes.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def posthoc_policy_generator_with_optimization() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a Python programming agent to reduce errors in solving user requirements.

        Given the following previous policy:
        {previous_policy}

        Here is a new error case that has occurred:
        {new_wrong_case}

        The effectiveness of the previous policy is measured by the following confidence score (Exponentially Weighted Moving Average accuracy):
        {confidence_score}

        - If the confidence score is high, only minor policy refinements may be needed.
        - If the confidence score is low, consider making more significant changes to the policy.

        Please analyze the new error case and revise the policy accordingly.
        Focus on Python code correctness and how to pass the required tests.
        Remove redundant or obsolete points. If certain previous policies are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise, actionable bullet points relevant to common Python coding and testing mistakes.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def posthoc_policy_generator_with_correct_cases() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a Python programming agent to reduce errors in solving user requirements.

        Given the following previous policy:
        {previous_policy}

        Here is a new error case that has occurred:
        {new_wrong_case}

        Here are some previous correct cases for reference:
        {previous_correct_case}

        Please analyze the new error case and revise the policy accordingly.
        Focus on Python code correctness and how to pass the required tests.
        Remove redundant or obsolete points. If certain previous policies are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise, actionable bullet points relevant to common Python coding and testing mistakes.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def posthoc_policy_generator_with_optimization_and_correct_cases() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a Python programming agent to reduce errors in solving user requirements.

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
        Focus on Python code correctness and how to pass the required tests.
        Remove redundant or obsolete points. If certain previous policies are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise, actionable bullet points relevant to common Python coding and testing mistakes.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def rag_policy_generator() -> str:
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a Python programming agent to reduce errors in solving user requirements.

        Given the following previous policy:
        {previous_policy}

        Here are several error cases that have occurred:
        {previous_wrong_cases}

        Please analyze all the new error cases and revise the policy accordingly.
        Focus on Python code correctness and how to pass the required tests.
        Remove redundant or obsolete points. If certain previous policies are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise, actionable bullet points relevant to common Python coding and testing mistakes.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def rag_policy_generator_with_optimization() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a Python programming agent to reduce errors in solving user requirements.

        Given the following previous policy:
        {previous_policy}

        Here are several error cases that have occurred:
        {previous_wrong_cases}

        The effectiveness of the previous policy is measured by the following confidence score (Exponentially Weighted Moving Average accuracy):
        {confidence_score}

        - If the confidence score is high, only minor policy refinements may be needed.
        - If the confidence score is low, consider making more significant changes to the policy.

        Please analyze all the new error cases and revise the policy accordingly.
        Focus on Python code correctness and how to pass the required tests.
        Remove redundant or obsolete points. If certain previous policies are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise, actionable bullet points relevant to common Python coding and testing mistakes.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def rag_policy_generator_with_correct_cases() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a Python programming agent to reduce errors in solving user requirements.

        Given the following previous policy:
        {previous_policy}

        Here are several error cases that have occurred:
        {previous_wrong_cases}

        Here are some previous correct cases for reference:
        {previous_correct_case}

        Please analyze all the new error cases and revise the policy accordingly.
        Focus on Python code correctness and how to pass the required tests.
        Remove redundant or obsolete points. If certain previous policies are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise, actionable bullet points relevant to common Python coding and testing mistakes.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def rag_policy_generator_with_optimization_and_correct_cases() -> str:
        """.."""
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a Python programming agent to reduce errors in solving user requirements.

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
        Focus on Python code correctness and how to pass the required tests.
        Remove redundant or obsolete points. If certain previous policies are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise, actionable bullet points relevant to common Python coding and testing mistakes.

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