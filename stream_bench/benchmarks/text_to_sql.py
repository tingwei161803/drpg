import copy
import os
import re
import textwrap

from colorama import Fore, Style
from datasets import Dataset
from stream_bench.benchmarks.base import Bench
from stream_bench.benchmarks.text2sql_utils.sqlite_interpreter import \
    execute_model
from stream_bench.benchmarks.text2sql_utils.string_formatter import (
    generate_schema_prompt, parse_sql)
from stream_bench.benchmarks.utils import strip_all_lines
from tqdm import tqdm


def create_cosql():
    class StreamingCOSQL(GeneralText2SQL):
        DATASET_PATH = 'appier-ai-research/StreamBench'
        DATASET_NAME = 'cosql'
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    return StreamingCOSQL

def create_spider():
    class StreamingSpider(GeneralText2SQL):
        DATASET_PATH = 'appier-ai-research/StreamBench'
        DATASET_NAME = 'spider'
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    return StreamingSpider


def create_bird():
    class StreamingBird(GeneralText2SQL):
        DATASET_PATH = 'appier-ai-research/StreamBench'
        DATASET_NAME = 'bird'
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    return StreamingBird


class GeneralText2SQL(Bench):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    NUM_SHOTS = 16
    instruction_template = """You are optimizing the policy for a text-to-SQL agent to reduce errors in generating correct SQL queries."""
    ref_previous_policy = """
        Given the following previous policy:
        {previous_policy}"""
    ref_new_wrong_case = """
        Here is a new error case that has occurred:
        {new_wrong_case}"""
    ref_rag_wrong_cases = """
        Here are several error cases that have occurred:
        {previous_wrong_cases}"""
    ref_rag_correct_cases = """
        Here are several correct cases that have occurred:
        {previous_correct_cases}"""
    rules_ver1 = """
        Please analyze all the cases and revise the policy accordingly.
        Focus on text-to-SQL correctness and how to pass the required tests.
        Remove redundant or obsolete points. If certain previous policy points are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise and actionable bullet points relevant to common text-to-SQL mistakes.

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
        {previous_cases}"""
    rules_ver1_dc = """
        Please analyze all the previous cases and revise the policy accordingly.
        Distill reusable strategies and common pitfalls for generating SQL queries from these cases.
        Remove redundant or obsolete points. If certain previous policy points are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise and actionable bullet points relevant to common text-to-SQL situations.

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
        knowledge: bool = False,
        db_path: str = None,
        agent = None,
        agent_callback = None,
        distribution_shift = False,
        **kwargs
    ) -> None:
        super().__init__(config={})
        self.split = split
        self.seed = seed
        self.feedback = feedback
        self.db_path = db_path
        self.feedback = feedback
        self.agent_callback = None
        self.knowledge = knowledge
        self.total = 0
        if distribution_shift:  # order the instances by sorting the field "db_id"
            print(Fore.CYAN + "Sorting the instances by 'db_id' for distribution shift evaluation." + Style.RESET_ALL)
            self.eval_set = self.dataset[self.split].sort("db_id")
        else:
            self.eval_set = self.dataset[self.split].shuffle(seed=self.seed)
        self.initialize()

    def get_dataset(self) -> Dataset:
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.eval_set

    def initialize(self) -> None:
        print("Initializing DB schema prompts...")
        self.db_prompt_schema = dict()
        for row in tqdm(self.eval_set, dynamic_ncols=True):
            if row["db_id"] not in self.db_prompt_schema:
                db_path = os.path.join(self.db_path, row["db_id"], row["db_id"] + ".sqlite")
                schema_prompt = generate_schema_prompt(db_path)
                self.db_prompt_schema[row["db_id"]] = schema_prompt
        print(len(self.db_prompt_schema), 'found!')
        print("Building few-shot examples...")
        self.fewshot_text = self.get_fewshot_text()

    def postprocess_generation(self, generation: str, idx: int = -1) -> str:
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for GeneralText2SQL-Task)
        """
        return parse_sql(generation)

    def get_input(self, row: dict) -> dict:
        """Builds the prompt for the LM to generate from."""
        row_input = copy.deepcopy(row)
        schema = self.db_prompt_schema[row["db_id"]]
        row_input["question"] = row["question"]
        row_input["fewshot_template"] = self.get_fewshot_template(schema=schema, question=row["question"])
        row_input["fewshotcot_template"] = self.get_fewshotcot_template(schema=schema, question=row["question"])
        row_input["prompt_zeroshot"] = self.get_zeroshot_prompt(schema=schema, question=row["question"])
        row_input["prompt_fewshot"] = self.get_fewshot_prompt(schema=schema, question=row["question"])
        row_input["prompt_cot"] = self.get_cot_prompt(schema=schema, question=row["question"])
        row_input["feedback_template"] = self.get_feedback_template(schema=schema, question=row["question"])
        row_input["refine_template"] = self.get_refine_template(schema=schema, question=row["question"])
        row_input["critic_feedback_template"] = self.get_critic_feedback_template(schema=schema, question=row["question"])
        row_input["critic_refine_template"] = self.get_critic_refine_template(schema=schema, question=row["question"])
        row_input["fewshot_template_with_policy"] = self.get_fewshot_template_with_policy(schema=schema, question=row["question"])
        row_input["prompt_zeroshot_with_policy"] = self.get_zeroshot_prompt_with_policy(schema=schema, question=row["question"])
        row_input["prompt_cot_with_policy"] = self.get_cot_prompt_with_policy(schema=schema, question=row["question"])
        row_input["fewshotcot_template_with_policy"] = self.get_fewshotcot_template_with_policy(schema=schema, question=row["question"])

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

    def get_output(self, row: dict):
        return {"SQL": row["SQL"], "db_id": row["db_id"], 'label': row["SQL"]}

    def process_results(self, generations: str, label: dict, return_details: bool = False, **kwargs):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param labels: original labels
            list of str containing refrences
        """
        db_path = os.path.join(self.db_path, label['db_id'], label['db_id'] + '.sqlite')
        correct = execute_model(generations, label['SQL'], db_path)[0]
        self.n_correct += correct
        self.predictions.append(generations)
        self.references.append(label)
        self.total += 1
        rolling_acc = self.n_correct / self.total
        if return_details:
            return {
                "result": 'Answer is Correct' if correct == 1 else 'Answer is NOT Correct',
                "correct": correct,
                "n_correct": self.n_correct,
                "rolling_acc": rolling_acc
            }
        return correct

    def get_metrics(self):
        return {
            "EX": self.n_correct / self.total
        }

    def give_feedback(self, model_output: str, row: dict, res: dict = None) -> tuple[bool, dict]:
        generation = self.postprocess_generation(model_output, -1)
        db_path = os.path.join(self.db_path, row['db_id'], row['db_id']+'.sqlite')
        correct, results = execute_model(generation, row['SQL'], db_path)
        # return (True, {"feedback_msg" : row['evidence'], "answer_correct": correct })
        pred_str = self.postprocess_generation(model_output)
        has_feedback = True
        feedbacks = {
            "question": row["question"],
            "self_output": pred_str,
            "is_correct": correct,
            "ground_truth": row["SQL"],
            "shot_template": self.get_shot_template(),
            "memprompt_template": self.get_memprompt_template()
        }
        return has_feedback, feedbacks

    @staticmethod
    def get_zeroshot_prompt(schema: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        {schema}

        -- Using valid SQLite, answer the following question for the tables provided above.
        -- Question: {question}

        Now, generate the correct SQL code directly (Do NOT generate other text except the SQL code):```sql\n<your_SQL_code>\n```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_cot_prompt(schema: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        {schema}

        -- Using valid SQLite, answer the following question for the tables provided above.
        -- Question: {question}

        Now, take a deep breath and work on this problem step-by-step to derive the correct SQL code.
        Provide your output in the following format:
        REASONING: <your_rationale>
        ANSWER: ```sql\n<your_SQL_code>\n```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template(schema: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing the text-to-SQL task. Here are some examples:

        {{fewshot_text}}

        Now it's your turn.

        -- SQL schema: {schema}
        -- Using valid SQLite, answer the following question for the SQL schema provided above.
        -- Question: {question}

        Now, generate the correct SQL code directly (Do NOT generate other text except the SQL code):```sql\n<your_SQL_code>\n```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshotcot_template(schema: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing the text-to-SQL task. Here are some examples:

        {{fewshot_text}}

        Now it's your turn.

        -- SQL schema: {schema}
        -- Using valid SQLite, answer the following question for the SQL schema provided above.
        -- Question: {question}

        Now, take a deep breath and work on this problem step-by-step to derive the correct SQL code.
        Provide your output in the following format:
        REASONING: <your_rationale>
        ANSWER: ```sql\n<your_SQL_code>\n```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_feedback_template(schema: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing the text-to-SQL task. Here is the database schema, user's question, and your previously generated SQL code.

        -- SQL schema: {schema}
        -- User's question: {question}
        -- Your SQL code: {{y_hat}}

        First, determine whether you need to refine your SQL code in terms of its correctness.
        If you consider that your SQL code is correct, output 'NO NEED TO REFINE' in uppercase.
        Otherwise, provide a suggestion to correct the SQL code.""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_refine_template(schema: str, question: str) -> str:
        """Note: The format of answer-feedback trajectory should be as follows

        Answer 0: <SQL_code_0>
        Feedback 0: <feedback_0>
        Answer 1: <SQL_code_1>
        Feedback 1: <feedback_1>
        ...
        Answer k: <SQL_code_n>
        Feedback k: <feedback_n>
        """
        prompt = textwrap.dedent(f"""\
        You are performing the text-to-SQL task. Here is the database schema, user's question, and your previous answer-feedback trajectory.

        -- SQL schema: {schema}
        -- User's question: {question}
        -- Your previous answer-feedback trajectory:
        {{trajectory}}

        According to the latest feedback, provide your refined SQL code.
        Provide your output in the following format: ```sql\n<refined_SQL_code>\n```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_critic_feedback_template(schema: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are an expert evaluator reviewing the correctness of SQL code generated for a text-to-SQL task. Consider the following database schema and user question:

        -- SQL schema: {schema}
        -- User's question: {question}
        -- Provided SQL code: {{y_hat}}

        Review the provided SQL code considering previous similar cases listed below:
        {{previous_cases}}

        Determine clearly:
        1. Whether the provided SQL code should be refined or not.
        2. Briefly explain the reason for your decision based on the provided schema, question, and previous cases.
        3. If refinement is needed, provide a suggestion to correct the SQL code.

        If no refinement is needed, output 'NO NEED TO REFINE' in uppercase.""")
        return strip_all_lines(prompt)


    @staticmethod
    def get_critic_refine_template(schema: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing the text-to-SQL task and need to refine your SQL code based on expert feedback. Consider the following database schema and user question:

        -- SQL schema: {schema}
        -- User's question: {question}
        -- Your previous SQL code: {{y_hat}}

        Feedback from expert evaluator:
        {{critic_rationale}}

        According to the feedback, provide your refined SQL code clearly formatted as: ```sql\n<refined_SQL_code>\n```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_shot_template() -> str:
        prompt = textwrap.dedent(f"""\
        Question: {{question}}
        {{answer}}""")
        return prompt

    @staticmethod
    def get_memprompt_template() -> str:
        prompt = textwrap.dedent(f"""\
        Question: {{question}}
        Your SQL code: {{answer}}
        User Feedback: {{correctness}}""")
        return prompt

    def get_fewshot_prompt(self, schema: str, question: str) -> str:
        fewshot_template = self.get_fewshot_template(schema, question)
        return re.sub(pattern=r"\{fewshot_text\}", repl=self.fewshot_text, string=fewshot_template)

    def get_fewshot_text(self) -> str:
        train_rows = self.dataset["train"].shuffle(seed=self.seed)
        shots = list()
        for i in range(self.NUM_SHOTS):
            shot = self.get_shot_template().format(
                question=train_rows[i]["question"],
                answer=train_rows[i]["SQL"]
            )
            shots.append(shot)
        return "\n\n\n".join(shots).replace("\\", "\\\\")

    @staticmethod
    def get_fewshot_template_with_policy(schema: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing the text-to-SQL task. Here are some examples:

        {{fewshot_text}}

        Please pay special attention to the following points, which are derived from previous cases.
        {{policy}}

        Now it's your turn.

        -- SQL schema: {schema}
        -- Using valid SQLite, answer the following question for the SQL schema provided above.
        -- Question: {question}

        Now, generate the correct SQL code directly (Do NOT generate other text except the SQL code):```sql\n<your_SQL_code>\n```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_zeroshot_prompt_with_policy(schema: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing the text-to-SQL task.

        Please pay special attention to the following points, which are derived from previous cases.
        {{policy}}

        Now it's your turn.

        -- SQL schema: {schema}
        -- Using valid SQLite, answer the following question for the SQL schema provided above.
        -- Question: {question}

        Now, generate the correct SQL code directly (Do NOT generate other text except the SQL code):```sql\n<your_SQL_code>\n```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshotcot_template_with_policy(schema: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing the text-to-SQL task. Here are some examples:

        {{fewshot_text}}

        Please pay special attention to the following points, which are derived from previous cases.
        {{policy}}

        Now it's your turn.

        -- SQL schema: {schema}
        -- Using valid SQLite, answer the following question for the SQL schema provided above.
        -- Question: {question}

        Now, take a deep breath and work on this problem step-by-step to derive the correct SQL code.
        Provide your output in the following format:
        REASONING: <your_rationale>
        ANSWER: ```sql\n<your_SQL_code>\n```""")
        return strip_all_lines(prompt)

    @staticmethod
    def get_cot_prompt_with_policy(schema: str, question: str) -> str:
        prompt = textwrap.dedent(f"""\
        You are performing the text-to-SQL task.

        Please pay special attention to the following points, which are derived from previous cases.
        {{policy}}

        Now it's your turn.

        -- SQL schema: {schema}
        -- Using valid SQLite, answer the following question for the SQL schema provided above.
        -- Question: {question}

        Now, take a deep breath and work on this problem step-by-step to derive the correct SQL code.
        Provide your output in the following format:
        REASONING: <your_rationale>
        ANSWER: ```sql\n<your_SQL_code>\n```""")
        return strip_all_lines(prompt)

    @staticmethod
    def posthoc_policy_generator() -> str:
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a text-to-SQL agent to reduce errors in generating correct SQL queries.

        Given the following previous policy:
        {previous_policy}

        Here is a new error case that has occurred:
        {new_wrong_case}

        Please analyze the new error case and revise the policy accordingly.
        Focus on common text-to-SQL mistakes, such as misunderstanding schema, incorrect SQL syntax, or invalid logic in the generated query.
        Remove redundant or obsolete points. If certain previous policy points are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise and actionable bullet points.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def posthoc_policy_generator_with_optimization() -> str:
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a text-to-SQL agent to reduce errors in generating correct SQL queries.

        Given the following previous policy:
        {previous_policy}

        Here is a new error case that has occurred:
        {new_wrong_case}

        The effectiveness of the previous policy is measured by the following confidence score (Exponentially Weighted Moving Average accuracy):
        {confidence_score}

        - If the confidence score is high, only minor policy refinements may be needed.
        - If the confidence score is low, consider making more significant changes to the policy.

        Please analyze the new error case and revise the policy accordingly.
        Focus on common text-to-SQL mistakes, such as misunderstanding schema, incorrect SQL syntax, or invalid logic in the generated query.
        Remove redundant or obsolete points. If certain previous policy points are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise and actionable bullet points.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def posthoc_policy_generator_with_correct_cases() -> str:
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a text-to-SQL agent to reduce errors in generating correct SQL queries.

        Given the following previous policy:
        {previous_policy}

        Here is a new error case that has occurred:
        {new_wrong_case}

        Here are some previous correct cases for reference:
        {previous_correct_case}

        Please analyze the new error case and revise the policy accordingly.
        Focus on common text-to-SQL mistakes, such as misunderstanding schema, incorrect SQL syntax, or invalid logic in the generated query.
        Remove redundant or obsolete points. If certain previous policy points are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise and actionable bullet points.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def posthoc_policy_generator_with_optimization_and_correct_cases() -> str:
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a text-to-SQL agent to reduce errors in generating correct SQL queries.

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
        Focus on common text-to-SQL mistakes, such as misunderstanding schema, incorrect SQL syntax, or invalid logic in the generated query.
        Remove redundant or obsolete points. If certain previous policy points are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise and actionable bullet points.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def rag_policy_generator() -> str:
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a text-to-SQL agent to reduce errors in generating correct SQL queries.

        Given the following previous policy:
        {previous_policy}

        Here are several error cases that have occurred:
        {previous_wrong_cases}

        Please analyze all the new error cases and revise the policy accordingly.
        Focus on common text-to-SQL mistakes, such as misunderstanding schema, incorrect SQL syntax, or invalid logic in the generated query.
        Remove redundant or obsolete points. If certain previous policy points are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise and actionable bullet points.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def rag_policy_generator_with_optimization() -> str:
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a text-to-SQL agent to reduce errors in generating correct SQL queries.

        Given the following previous policy:
        {previous_policy}

        Here are several error cases that have occurred:
        {previous_wrong_cases}

        The effectiveness of the previous policy is measured by the following confidence score (Exponentially Weighted Moving Average accuracy):
        {confidence_score}

        - If the confidence score is high, only minor policy refinements may be needed.
        - If the confidence score is low, consider making more significant changes to the policy.

        Please analyze all the new error cases and revise the policy accordingly.
        Focus on common text-to-SQL mistakes, such as misunderstanding schema, incorrect SQL syntax, or invalid logic in the generated query.
        Remove redundant or obsolete points. If certain previous policy points are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise and actionable bullet points.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def rag_policy_generator_with_correct_cases() -> str:
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a text-to-SQL agent to reduce errors in generating correct SQL queries.

        Given the following previous policy:
        {previous_policy}

        Here are several error cases that have occurred:
        {previous_wrong_cases}

        Here are some previous correct cases for reference:
        {previous_correct_case}

        Please analyze all the new error cases and revise the policy accordingly.
        Focus on common text-to-SQL mistakes, such as misunderstanding schema, incorrect SQL syntax, or invalid logic in the generated query.
        Remove redundant or obsolete points. If certain previous policy points are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise and actionable bullet points.

        Example output format:
        POLICY:
        - [first point]
        - [second point]""")
        return strip_all_lines(prompt)

    @staticmethod
    def rag_policy_generator_with_optimization_and_correct_cases() -> str:
        prompt = textwrap.dedent("""\
        You are optimizing the policy for a text-to-SQL agent to reduce errors in generating correct SQL queries.

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
        Focus on common text-to-SQL mistakes, such as misunderstanding schema, incorrect SQL syntax, or invalid logic in the generated query.
        Remove redundant or obsolete points. If certain previous policy points are still valid, keep them.

        Output the revised policy, starting with 'POLICY:' on a new line, followed immediately by a **bulleted list** (each point on a new line, starting with '- ').
        Limit the revised policy to at most 5 concise and actionable bullet points.

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

# Manual testing
if __name__ == "__main__":
    DB_ROOT_PATH = "../bird-benchmark/data/bird-benchmark/dev/dev_databases"
    DB_ID = "financial"
    db_path = os.path.join(DB_ROOT_PATH, DB_ID, DB_ID + ".sqlite")
    schema_prompt = generate_schema_prompt(db_path)
    # Test get_zeroshot_prompt
    question = "What is the total amount of money spent on all transactions?"
    zeroshot_prompt = GeneralText2SQL.get_zeroshot_prompt(schema_prompt, question)
    # Test get_fewshot_template
    fewshot_text = "SELECT SUM(amount) FROM transactions"
    fewshot_prompt = GeneralText2SQL.get_fewshot_template(schema_prompt, question).format(fewshot_text=fewshot_text)
    print(fewshot_prompt)
