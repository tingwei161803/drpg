from .base import Agent
from .dynamic_cheatsheet import DynamicCheatsheetAgent
from .fewshot_rag import FewShotRAGAgent
from .gt import GroundTruthAgent
from .iter_prompt import IterPromptAgent
from .policy import PolicyAgent
from .policy_rag import PolicyRAGAgent
from .zeroshot import ZeroShotAgent

classes = locals()

# Mapping from the `agent_name` field of an agent config to its implementation.
# The names below are the ones used in the paper's experiment logs:
#   rag_policy          -> DRPG (Ours)
#   rag_policy_rag      -> DRPG with a separate policy generator (cross-generator ablation)
#   dynamic_cheatsheet  -> Dynamic Cheatsheet, i.e. DRPG without environment feedback
#   self_stream_icl     -> Self-StreamICL (main baseline)
#   self_refine         -> Self-Refine
#   zeroshot            -> Zero-shot
#   mem_prompt          -> MemPrompt (StreamBench baseline, not reported in the paper)
#   gt                  -> Ground-truth oracle, used to sanity-check the environment
TASKS = {
    "zeroshot": ZeroShotAgent,
    "self_refine": IterPromptAgent,
    "mem_prompt": FewShotRAGAgent,
    "self_stream_icl": FewShotRAGAgent,
    "rag_policy": PolicyAgent,
    "rag_policy_rag": PolicyRAGAgent,
    "dynamic_cheatsheet": DynamicCheatsheetAgent,
    "gt": GroundTruthAgent,
}


def load_agent(agent_name):
    if agent_name in TASKS:
        return TASKS[agent_name]
    if agent_name in classes:
        return classes[agent_name]

    raise ValueError("Agent %s not found" % agent_name)
