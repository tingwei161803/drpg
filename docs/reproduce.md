# Reproducing the paper

Every result in the paper comes from one invocation of `run_bench.py` with one agent config and one benchmark config. This page maps the tables to the configs in this repository.

## The grid

**Table 1, main results.** Four methods × 7 models × 6 benchmarks = 168 runs.

```bash
for method in zeroshot self_refine self_stream_icl drpg; do
  for model in gemini-2.0-flash gemini-2.0-flash-lite llama-3.3-70b \
               llama-4-maverick llama-4-scout mistral-medium mistral-small; do
    for bench in spider cosql bird hotpotqa ddxplus ds_1000; do
      uv run python -m stream_bench.pipelines.run_bench \
        --agent_cfg "configs/agent/$method/$model.yml" \
        --bench_cfg "configs/bench/$bench.yml"
    done
  done
done
```

Run them one at a time: each one is a stream, and the point of the benchmark is that step *t* depends on steps *1..t-1*.

| Result | Configs |
| --- | --- |
| Table 1, main results | `configs/agent/{zeroshot,self_refine,self_stream_icl,drpg}/<model>.yml` |
| Table 4 (Sec. 5.3), cross-generator | `configs/agent/drpg-cross-generator/<agent>__policy-<generator>.yml`, plus the matching `drpg/<model>.yml` for the same-model rows |
| Appendix D, retrieval strategy | `drpg/<model>.yml` with `policy_generator_mode` set to `correct` or `wrong` |
| Appendix E, previous policy | `drpg/<model>.yml` with `use_previous_policy: true` |
| Appendix F, no environment feedback | `configs/agent/dynamic_cheatsheet/<model>.yml` |
| Appendix H, Qwen and Gemma | add the model to `MODELS` in `scripts/gen_configs.py` (see below) |

For the ablations, edit the corresponding field in `scripts/gen_configs.py`, regenerate into a scratch directory, and point `--agent_cfg` at it:

```bash
uv run scripts/gen_configs.py --out /tmp/ablation-configs
```

## The models

| Slug in this repo | `series` | `model_name` |
| --- | --- | --- |
| `gemini-2.0-flash` | `gemini_dev` | `gemini-2.0-flash` |
| `gemini-2.0-flash-lite` | `gemini_dev` | `gemini-2.0-flash-lite` |
| `llama-3.3-70b` | `nvidia` | `meta/llama-3.3-70b-instruct` |
| `llama-4-maverick` | `nvidia` | `meta/llama-4-maverick-17b-128e-instruct` |
| `llama-4-scout` | `nvidia` | `meta/llama-4-scout-17b-16e-instruct` |
| `mistral-medium` | `nvidia` | `mistralai/mistral-medium-3-instruct` |
| `mistral-small` | `nvidia` | `mistralai/mistral-small-3.1-24b-instruct-2503` |

Appendix H additionally reports `qwen/qwen3.5-122b-a10b` and `google/gemma-4-31b-it`, both served through NVIDIA NIM and both run without thinking: `stream_bench/llms/nvidia_api.py` explicitly turns thinking off for Qwen hybrids, which otherwise default to having it on. Note that endpoints come and go: `qwen/qwen3.5-122b-a10b` was withdrawn from NIM after the paper's experiments and can no longer be reached.

All experiments ran inside the free tiers of Google AI Studio, NVIDIA NIM and the Meta Llama API, so the cost of reproducing a cell is the provider's rate limit rather than money. If you get throttled, add `--slow_task` (5 s between steps) or `--slow_slow_task` (10 s).

## What to expect

These are hosted models behind APIs. Served checkpoints are updated, quantized and retired without notice, and a rerun months later is not the same experiment. Expect the *ordering* to reproduce rather than the exact decimals: DRPG over Self-StreamICL over Self-Refine and Zero-shot on most text-to-SQL and diagnosis configurations, with DS-1000 and HotpotQA the tasks where policy-level guidance helps least.

Two details worth knowing before comparing numbers:

- **Cost is in tokens, not calls.** `num_inference_call` in the logs is hard-coded to 1; DRPG makes two real calls per query. The token counters already sum both.
- **Different runs, same stream.** `seed: 42` fixes the order, so any two methods see identical inputs in identical order. This is what makes the per-configuration comparisons in Appendix G meaningful.

## Analysis

The paper's tables and figures were produced from the Weights & Biases logs of these runs. Passing `--use_wandb --entity <your entity>` reproduces that logging structure: one project per benchmark (`streambench-<bench_name>`), one run per configuration, with `config.agent_name` identifying the method and `final/<metric>` carrying the score.
