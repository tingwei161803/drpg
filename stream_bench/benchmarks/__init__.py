from .base import Bench
from .ddxplus import create_ddxplus
from .ds_1000 import DS1000
from .hotpotqa_distract import HotpotQADistract
from .text_to_sql import create_bird, create_cosql, create_spider

classes = locals()

# The six benchmarks used in the paper. `bench_name` in a benchmark config
# selects one of these.
TASKS = {
    "spider": create_spider(),
    "cosql": create_cosql(),
    "bird": create_bird(),
    "hotpotqa_distract": HotpotQADistract,
    "ddxplus": create_ddxplus(),
    "ds_1000": DS1000,
}


def load_benchmark(benchmark_name) -> Bench:
    if benchmark_name in TASKS:
        return TASKS[benchmark_name]
    if benchmark_name in classes:
        return classes[benchmark_name]

    raise ValueError("Benchmark %s not found" % benchmark_name)
