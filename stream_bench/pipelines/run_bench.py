import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import wandb
import yaml
from stream_bench.agents import load_agent
from stream_bench.benchmarks import Bench, load_benchmark
from tqdm import tqdm

from .utils import merge_dicts


def setup_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--agent_cfg",
        type=Path,
        required=True,
        help="Path to the agent's config yaml file"
    )
    parser.add_argument(
        "--bench_cfg",
        type=Path,
        required=True,
        help="Path to the benchmark's config yaml file."
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use wandb for experiment tracking."
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project name for wandb"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Team name for wandb"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="The name of the experiment for wandb."
    )
    parser.add_argument(
        "--slow_task",
        action="store_true",
        help="Whether to slow down each task with longer sleep time between iterations."
    )
    parser.add_argument(
        "--slow_slow_task",
        action="store_true",
        help="Whether to slow down each task with even longer sleep time between iterations."
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Notes for this experiment (will show in wandb run)."
    )
    return parser.parse_args()


def main():
    args = setup_args()
    agent_cfg = yaml.safe_load(args.agent_cfg.read_text())
    bench_cfg = yaml.safe_load(args.bench_cfg.read_text())
    assert 'bench_name' in bench_cfg
    agent_cfg["bench_name"] = bench_cfg["bench_name"]
    agent_cfg["split"] = bench_cfg["split"]
    if args.name is not None:
        agent_cfg['exp_name'] = args.name
    print('init agent')
    if bench_cfg['bench_name'] == "bigcodebench" or bench_cfg['bench_name'] == "ds_1000" or bench_cfg['bench_name'] == "hotpotqa_distract":
        if 'rag' in agent_cfg:
            agent_cfg['rag']['top_k'] = 4
    agent = load_agent(agent_cfg['agent_name'])(agent_cfg)
    bench_cfg['agent'] = agent
    # bench_cfg['agent_callback'] = agent.retrieve_experience
    print('init bench environment')
    bench: Bench = load_benchmark(bench_cfg['bench_name'])(**bench_cfg)
    agent.bench = bench

    # Get total number of tasks for model switching
    dataset_list = list(bench.get_dataset())
    total_tasks = len(dataset_list)
    half_point = total_tasks // 2

    if args.use_wandb:
        # Prepare config for wandb
        wandb_config = merge_dicts(dicts=[agent_cfg, bench_cfg])

        # Add critic_llm info if available
        if hasattr(agent, 'critic_llm_config') and agent.critic_llm_config:
            wandb_config['critic_llm'] = agent.critic_llm_config
        else:
            wandb_config['critic_llm'] = None

        wandb.init(
            project=args.project if (args.project is not None) else f"streambench-{bench_cfg['bench_name']}",
            entity=args.entity,
            name=args.name if (args.name is not None) else agent.get_name(),
            config=wandb_config,
            notes=args.notes
        )

    for time_step, row in enumerate(tqdm(dataset_list, dynamic_ncols=True)):
        try:
            # Check if we need to switch models at halfway point
            if (agent_cfg.get("enabled_transfer", False) and
                time_step == half_point and
                hasattr(agent, 'llm2') and
                agent.llm2 is not None):
                print(f"\nSwitching models at time step {time_step}")
                agent.switch_models()

            row['time_step'] = time_step
            x = bench.get_input(row)  # remove ground truth related information
            x['time_step'] = time_step
            model_output = agent(**x)
            prediction = bench.postprocess_generation(model_output, time_step)
            label = bench.get_output(row)
            pred_res = bench.process_results(
                prediction,
                label,
                return_details=True,
                time_step=time_step
            )

            has_feedback, feedback = bench.give_feedback(model_output, row, pred_res)
            feedback['time_step'] = time_step
            feedback = { **row, **feedback }
            agent.update(has_feedback, **feedback)

            if args.use_wandb:
                log_data = merge_dicts(dicts=[agent.get_wandb_log_info(), pred_res])
                wandb.log(data=log_data)
            # NOTE: agent.log() should be called after wandb
            if isinstance(label, int):
                label = bench.LABEL2TEXT[label]
            elif isinstance(label, dict):
                label = label.get("label", json.dumps(label))

            agent.log(label_text=label)
        except (KeyError, IndexError) as e:
            print(e)

        # Sleep between steps to stay within API rate limits; --slow_task waits longer
        if args.slow_task:
            time.sleep(5)  # --slow_task
        elif args.slow_slow_task:
            time.sleep(10)
        # Default pacing
        else:
            time.sleep(1)   # default

    metrics = bench.get_metrics()
    print(metrics)
    if args.use_wandb:
        wandb.log(data={'final/'+k: v for k, v in metrics.items()})

if __name__ == "__main__":
    main()
