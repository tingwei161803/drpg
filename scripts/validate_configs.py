# /// script
# requires-python = ">=3.10"
# dependencies = ["PyYAML", "colorama"]
# ///
"""Instantiate every agent config without touching the network.

The LLM backends and the FAISS-backed memory are replaced by stubs, so this
checks the thing that actually breaks in practice: a config that is missing a
key an agent reads in ``__init__`` or ``get_name``. Run it after editing
``scripts/gen_configs.py`` or after adding a model.

    uv run scripts/validate_configs.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The retrieval stack and the dataset loaders are imported at module level but
# never exercised here, so stub them out: this check should run in a second
# without a multi-gigabyte install.
for _heavy in ("faiss", "torch", "transformers", "numpy", "datasets", "evaluate",
               "pandas", "tqdm", "func_timeout"):
    sys.modules.setdefault(_heavy, mock.MagicMock(name=_heavy))


class _StubLLM:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def __call__(self, prompt: str, **kwargs):  # pragma: no cover - never called
        raise AssertionError("the stub LLM must not be called during validation")


class _StubRAG:
    def __init__(self, config: dict) -> None:
        self.top_k = config["top_k"]
        self.retrieve_order = config["order"]
        self.insert_acc = 0


def main() -> int:
    from stream_bench.agents import load_agent

    configs = sorted((ROOT / "configs" / "agent").rglob("*.yml"))
    if not configs:
        print("no agent configs found -- run scripts/gen_configs.py first")
        return 1

    benches = sorted((ROOT / "configs" / "bench").glob("*.yml"))
    bench_names = [yaml.safe_load(p.read_text())["bench_name"] for p in benches]

    failures = []
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        # Agents open a jsonlines log under ./log on construction; keep that in
        # a temporary directory instead of the working tree.
        os.chdir(tmp)
        patches = [
            mock.patch("stream_bench.agents.base.get_llm",
                       lambda series, model_name: _StubLLM(model_name)),
            mock.patch("stream_bench.agents.utils.RAG", _StubRAG),
            mock.patch("stream_bench.agents.utils_rag.RAG", _StubRAG),
            mock.patch("stream_bench.agents.policy.RAG", _StubRAG),
            mock.patch("stream_bench.agents.policy_rag.RAG", _StubRAG),
            mock.patch("stream_bench.agents.fewshot_rag.RAG", _StubRAG),
            mock.patch("stream_bench.agents.dynamic_cheatsheet.RAG", _StubRAG),
        ]
        for patch in patches:
            patch.start()
        try:
            for path in configs:
                config = yaml.safe_load(path.read_text())
                # run_bench.py copies these two fields from the benchmark config.
                config["bench_name"] = bench_names[0]
                config["split"] = "test"
                config["exp_name"] = "validate"
                try:
                    agent = load_agent(config["agent_name"])(config)
                    agent.get_name()
                except Exception as exc:  # noqa: BLE001 - report, do not raise
                    failures.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
        finally:
            for patch in patches:
                patch.stop()
            os.chdir(cwd)

    for failure in failures:
        print("FAIL", failure)
    print(f"{len(configs) - len(failures)}/{len(configs)} agent configs instantiate "
          f"({len(benches)} benchmark configs: {', '.join(bench_names)})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
