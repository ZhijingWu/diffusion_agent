import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from experiments.audits.canonical_runtime_adapter import (
    canonicalize_for_current_runtime,
)

from GDesigner.graph.graph import Graph
from GDesigner.gdt.gtd_framework import GTDFramework
from GDesigner.gdt.proxy_reward_model import ProxyRewardModel
from GDesigner.gdt.guided_generation import GuidedGeneration
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Matched control
#
# Same K Bernoulli candidates as official guidance,
# but completely ignore Proxy reward and always select
# candidate 0.
# ============================================================

class FirstCandidateGuider:

    def __init__(self, num_candidates_per_step: int = 5):
        self.num_candidates_per_step = num_candidates_per_step

    @torch.no_grad()
    def guide(
        self,
        current_At_prob: torch.Tensor,
        timestep: torch.Tensor,
        unguided_A0_prediction: torch.Tensor,
        node_features: torch.Tensor,
        task_condition: torch.Tensor,
    ):

        batch_size, _, _ = unguided_A0_prediction.shape

        output = torch.zeros_like(
            unguided_A0_prediction
        )

        for i in range(batch_size):

            probs = unguided_A0_prediction[i]

            # IMPORTANT:
            # Sample exactly K candidates,
            # matching official GuidedGeneration.
            candidates = [
                torch.bernoulli(
                    probs
                ).float()
                for _ in range(
                    self.num_candidates_per_step
                )
            ]

            # Matched stochastic control:
            # do NOT evaluate Proxy.
            output[i] = candidates[0]

        return output


# ============================================================
# Helpers
# ============================================================

def scalar_value(singleton):
    """
    Read GDesigner global counter value safely.
    """

    try:

        value = singleton.instance().value

        if value is None:
            return 0.0

        return float(value)

    except Exception:

        return 0.0


def edge_count(matrix):

    return int(
        sum(
            sum(
                int(x)
                for x in row
            )
            for row in matrix
        )
    )


def save_json(path: Path, data):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Model loading
# ============================================================

def build_proxy(args, device):

    proxy_model = ProxyRewardModel(
        task_cond_input_dim=(
            args.task_cond_input_dim
        ),
        node_feature_dim=(
            args.node_feature_dim
        ),
        condition_dim=(
            args.condition_dim
        ),
        gnn_hidden_dim=(
            args.gnn_hidden_dim
        ),
        gnn_layers=(
            args.gnn_layers
        ),
        mlp_hidden_dim=(
            args.mlp_hidden_dim
        ),
        num_reward_components=2,
    ).to(device)

    state = torch.load(
        args.proxy_model,
        map_location=device,
    )

    proxy_model.load_state_dict(
        state
    )

    proxy_model.eval()

    proxy_model.reward_component_names = [
        "utility",
        "cost",
    ]

    return proxy_model


def build_diffusion_framework(
    args,
    device,
):

    framework = GTDFramework(
        task_cond_input_dim=(
            args.task_cond_input_dim
        ),
        node_feature_dim=(
            args.node_feature_dim
        ),
        condition_dim=(
            args.condition_dim
        ),
        time_embed_dim=(
            args.time_emb_dim
        ),
        gt_num_layers=(
            args.layers
        ),
        gt_num_heads=(
            args.heads
        ),
        diffusion_num_timesteps=(
            args.diffusion_steps
        ),
        device=device,
    )

    state = torch.load(
        args.diffusion_model,
        map_location=device,
    )

    framework.diffusion_model.load_state_dict(
        state
    )

    framework.diffusion_model.to(
        device
    )

    framework.diffusion_model.eval()

    return framework


# ============================================================
# Experiment
# ============================================================

async def run(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("GTD MATCHED DOWNSTREAM CONTROL")
    print("=" * 80)

    print("Arm:", args.arm)
    print("Device:", device)
    print("Dataset:", args.dataset)
    print("Limit:", args.limit)
    print("Base seed:", args.base_seed)
    print("Candidates K:", args.candidates)

    # --------------------------------------------------------
    # Load GSM8K
    # --------------------------------------------------------

    dataset = JSONLReader.parse_file(
        args.dataset
    )

    dataset = gsm_data_process(
        dataset
    )

    if args.limit is not None:

        dataset = dataset[
            :args.limit
        ]

    # --------------------------------------------------------
    # Agent team
    # --------------------------------------------------------

    agent_names_list = [

        name

        for name, num
        in zip(
            args.agent_names,
            args.agent_nums,
        )

        for _ in range(num)
    ]

    num_nodes = len(
        agent_names_list
    )

    prompt_set = (
        PromptSetRegistry.get(
            args.domain
        )
    )

    agent_profiles = [

        prompt_set.get_description(
            name
        )

        for name
        in agent_names_list
    ]

    # Exactly same semantic node features
    # as official GSM8K pipeline.
    node_features_np = np.stack(

        [
            np.asarray(
                get_sentence_embedding(
                    profile
                ),
                dtype=np.float32,
            )

            for profile
            in agent_profiles
        ],

        axis=0,
    )

    node_features_base = (
        torch
        .from_numpy(
            node_features_np
        )
        .to(device)
    )

    # --------------------------------------------------------
    # Diffusion model
    # --------------------------------------------------------

    framework = (
        build_diffusion_framework(
            args,
            device,
        )
    )

    # --------------------------------------------------------
    # Guider
    # --------------------------------------------------------

    if args.arm == "proxy":

        proxy_model = build_proxy(
            args,
            device,
        )

        guider = GuidedGeneration(
            proxy_reward_model=(
                proxy_model
            ),
            macp_weights={
                "utility": 1.0,
                "cost": -0.1,
            },
            num_candidates_per_step=(
                args.candidates
            ),
            device=device,
        )

    elif args.arm == "first":

        guider = FirstCandidateGuider(
            num_candidates_per_step=(
                args.candidates
            )
        )

    else:

        raise ValueError(
            f"Unknown arm: "
            f"{args.arm}"
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    timestamp = time.strftime(
        "%Y-%m-%d-%H-%M-%S",
        time.localtime(),
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (

        output_dir

        / (
            f"{args.arm}_candidate_"
            f"{args.llm_name}_"
            f"seed{args.base_seed}_"
            f"{timestamp}.json"
        )
    )

    results = []

    total_solved = 0
    total_executed = 0

    # ========================================================
    # Main loop
    # ========================================================

    for idx, record in enumerate(
        dataset
    ):

        task_query = record[
            "task"
        ]

        true_answer = record[
            "answer"
        ]

        # Important:
        # exactly same seed schedule
        # between proxy and first arms.
        seed = (
            args.base_seed
            + idx
        )

        print()
        print("-" * 80)

        print(
            f"[{idx + 1}/"
            f"{len(dataset)}] "
            f"arm={args.arm} "
            f"seed={seed}"
        )

        print(
            task_query[:100]
        )

        # ----------------------------------------------------
        # Seed topology generation
        # ----------------------------------------------------

        set_seed(seed)

        condition_np = np.asarray(

            get_sentence_embedding(
                task_query
            ),

            dtype=np.float32,
        )

        task_condition = (

            torch
            .from_numpy(
                condition_np
            )
            .unsqueeze(0)
            .to(device)
        )

        # ----------------------------------------------------
        # Generate topology
        # ----------------------------------------------------

        with torch.no_grad():

            generated_probs = (
                framework
                .diffusion_model
                .sample(
                    num_nodes=num_nodes,
                    batch_size=1,

                    node_features=(
                        node_features_base
                        .unsqueeze(0)
                    ),

                    task_condition=(
                        task_condition
                    ),

                    guider=guider,
                )
            )

        generated_adj = (

            generated_probs
            .squeeze(0)
            .gt(0.5)
            .int()
            .cpu()
        )

        generated_adj_list = (
            generated_adj.tolist()
        )

        # ----------------------------------------------------
        # Real MAS Graph
        # ----------------------------------------------------

        gdesigner_graph = Graph(
            domain=args.domain,
            llm_name=args.llm_name,
            agent_names=(
                agent_names_list
            ),
            decision_method=(
                args.decision_method
            ),
            fixed_spatial_masks=(
                canonicalize_for_current_runtime(
                    generated_adj_list
                )
            ),
        )

        # ----------------------------------------------------
        # Token / cost counters
        # ----------------------------------------------------

        cost_before = (
            scalar_value(Cost)
        )

        prompt_before = (
            scalar_value(
                PromptTokens
            )
        )

        completion_before = (
            scalar_value(
                CompletionTokens
            )
        )

        # ----------------------------------------------------
        # Run real MAS
        # ----------------------------------------------------

        raw_answer, _ = (
            await
            gdesigner_graph.arun(
                {
                    "task":
                    task_query
                },
                args.num_rounds,
            )
        )

        # ----------------------------------------------------
        # Counter deltas
        # ----------------------------------------------------

        cost_after = (
            scalar_value(Cost)
        )

        prompt_after = (
            scalar_value(
                PromptTokens
            )
        )

        completion_after = (
            scalar_value(
                CompletionTokens
            )
        )

        # ----------------------------------------------------
        # GSM8K evaluation
        # ----------------------------------------------------

        predict_answer = (
            gsm_get_predict(
                raw_answer[0]
            )
        )

        is_solved = (

            float(
                predict_answer
            )

            ==

            float(
                true_answer
            )

            if predict_answer

            else False
        )

        total_solved += int(
            is_solved
        )

        total_executed += 1

        accuracy = (
            total_solved
            / total_executed
        )

        # ----------------------------------------------------
        # Read actual executed graph AFTER cycle filtering
        # ----------------------------------------------------

        executed_adj_np = (
            gdesigner_graph
            .spatial_adj_matrix
        )

        executed_adj_list = (
            executed_adj_np
            .astype(int)
            .tolist()
        )

        raw_edges = edge_count(
            generated_adj_list
        )

        executed_edges = edge_count(
            executed_adj_list
        )

        # ----------------------------------------------------
        # Save row
        # ----------------------------------------------------

        row = {

            "Index": idx,

            "Seed": seed,

            "Arm": args.arm,

            "Question":
                task_query,

            "Answer":
                true_answer,

            "Response":
                raw_answer,

            "Attempt answer":
                predict_answer,

            "Solved":
                bool(
                    is_solved
                ),

            "Generated_Topology":
                generated_adj_list,

            "Executed_Topology":
                executed_adj_list,

            "Raw_Edges":
                raw_edges,

            "Executed_Edges":
                executed_edges,

            "Edges_Removed_By_Execution":
                (
                    raw_edges
                    - executed_edges
                ),

            "PromptTokens_Delta":
                (
                    prompt_after
                    - prompt_before
                ),

            "CompletionTokens_Delta":
                (
                    completion_after
                    - completion_before
                ),

            "TotalTokens_Delta":
                (
                    prompt_after
                    - prompt_before
                    +
                    completion_after
                    - completion_before
                ),

            "Cost_Delta":
                (
                    cost_after
                    - cost_before
                ),

            "Total solved":
                total_solved,

            "Total executed":
                total_executed,

            "Accuracy":
                accuracy,
        }

        results.append(
            row
        )

        save_json(
            output_path,
            results,
        )

        print(
            f"Solved="
            f"{is_solved}"
            f" | Accuracy="
            f"{accuracy:.3f}"
            f" | raw_edges="
            f"{raw_edges}"
            f" | executed_edges="
            f"{executed_edges}"
            f" | tokens="
            f"{row['TotalTokens_Delta']:.0f}"
        )

    # ========================================================
    # Summary
    # ========================================================

    if results:

        avg_raw = np.mean(
            [
                r["Raw_Edges"]
                for r in results
            ]
        )

        avg_exec = np.mean(
            [
                r["Executed_Edges"]
                for r in results
            ]
        )

        avg_tokens = np.mean(
            [
                r["TotalTokens_Delta"]
                for r in results
            ]
        )

        total_tokens = np.sum(
            [
                r["TotalTokens_Delta"]
                for r in results
            ]
        )

        unique_raw = len(
            {
                tuple(
                    tuple(row)
                    for row
                    in r[
                        "Generated_Topology"
                    ]
                )

                for r
                in results
            }
        )

        unique_exec = len(
            {
                tuple(
                    tuple(row)
                    for row
                    in r[
                        "Executed_Topology"
                    ]
                )

                for r
                in results
            }
        )

    else:

        avg_raw = 0
        avg_exec = 0
        avg_tokens = 0
        total_tokens = 0
        unique_raw = 0
        unique_exec = 0

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        "Arm:",
        args.arm,
    )

    print(
        "Records:",
        len(results),
    )

    print(
        "Accuracy:",
        (
            total_solved
            / total_executed
            if total_executed
            else 0
        ),
    )

    print(
        "Average raw edges:",
        round(
            float(avg_raw),
            4,
        ),
    )

    print(
        "Average executed edges:",
        round(
            float(avg_exec),
            4,
        ),
    )

    print(
        "Unique raw topologies:",
        unique_raw,
    )

    print(
        "Unique executed topologies:",
        unique_exec,
    )

    print(
        "Average token delta:",
        round(
            float(avg_tokens),
            2,
        ),
    )

    print(
        "Total token delta:",
        round(
            float(total_tokens),
            2,
        ),
    )

    print(
        "Saved:",
        output_path,
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Matched downstream "
            "GTD Proxy-vs-First "
            "candidate control."
        )
    )

    parser.add_argument(
        "--arm",
        choices=[
            "proxy",
            "first",
        ],
        required=True,
    )

    parser.add_argument(
        "--dataset",
        default=(
            "datasets/gsm8k/"
            "gsm8k_test_100.jsonl"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--proxy-model",
        default=(
            "proxy_model_gsm8k.pth"
        ),
    )

    parser.add_argument(
        "--diffusion-model",
        default=(
            "diffusion_model_gsm8k.pth"
        ),
    )

    parser.add_argument(
        "--llm-name",
        default="gpt-4o-mini",
    )

    parser.add_argument(
        "--domain",
        default="gsm8k",
    )

    parser.add_argument(
        "--decision-method",
        default="FinalRefer",
    )

    parser.add_argument(
        "--agent-names",
        nargs="+",
        default=[
            "MathSolver"
        ],
    )

    parser.add_argument(
        "--agent-nums",
        nargs="+",
        type=int,
        default=[4],
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=20261002,
    )

    # --------------------------------------------------------
    # Released/default GTD dimensions
    # --------------------------------------------------------

    parser.add_argument(
        "--task-cond-input-dim",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--node-feature-dim",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--condition-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--time-emb-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--gnn-hidden-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--gnn-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "audit_results/"
            "downstream_matched_control"
        ),
    )

    args = parser.parse_args()

    if (
        len(args.agent_names)
        != len(args.agent_nums)
    ):

        parser.error(
            "--agent-names and "
            "--agent-nums must "
            "have the same length"
        )

    return args


if __name__ == "__main__":

    asyncio.run(
        run(
            parse_args()
        )
    )