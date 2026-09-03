"""Minimal on-policy GRPO training script for GSM8K."""

import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import grpo_train_step
from cs336_alignment.vllm_utils import VLLMServer


# Change these values for different runs.
MODEL_ID = "allenai/OLMo-2-0425-1B"
PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
TRAIN_PATH = "data/gsm8k/train.jsonl"
VAL_PATH = "data/gsm8k/test.jsonl"

N_TRAIN_EXAMPLES = 6400
N_VAL_EXAMPLES = 1024
NUM_ROLLOUT_STEPS = int(os.environ.get("GRPO_STEPS", "50"))
ROLLOUT_BATCH_SIZE = 256
GROUP_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get("GRPO_GRAD_ACCUM", "64")
)
LEARNING_RATE = 1e-5
TEMPERATURE = 1.0
MAX_TOKENS = 512
MAX_GRAD_NORM = 1.0
EVAL_EVERY = 10
SEED = int(os.environ.get("GRPO_SEED", "42"))

# Fixed denominator for the constant-normalized variants.
NORMALIZATION_CONSTANT = int(
    os.environ.get(
        "GRPO_NORMALIZATION_CONSTANT",
        str(ROLLOUT_BATCH_SIZE * MAX_TOKENS),
    )
)

VARIANT = os.environ.get("GRPO_VARIANT", "standard")
VARIANT_CONFIGS = {
    "standard": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "normalization_constant": None,
    },
    "grpo_constant": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "constant",
        "normalization_constant": NORMALIZATION_CONSTANT,
    },
    "dr_grpo": {
        "baseline": "mean",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
        "normalization_constant": NORMALIZATION_CONSTANT,
    },
    "rft": {
        "baseline": "none",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
        "normalization_constant": NORMALIZATION_CONSTANT,
    },
    "maxrl": {
        "baseline": "mean",
        "advantage_normalizer": "mean",
        "loss_normalization": "constant",
        "normalization_constant": NORMALIZATION_CONSTANT,
    },
}
if VARIANT not in VARIANT_CONFIGS:
    raise ValueError(f"Unsupported GRPO_VARIANT: {VARIANT}")
VARIANT_CONFIG = VARIANT_CONFIGS[VARIANT]

OUTPUT_DIR = Path(
    os.environ.get("GRPO_OUTPUT_DIR", f"experiments/grpo_{VARIANT}")
)

POLICY_DEVICE = "cuda:0"
VLLM_GPU = 2
VLLM_PORT = 8010


def load_examples(path: str, limit: int) -> list[dict[str, str]]:
    examples = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            answer = item["answer"].split("####", 1)[1].strip()
            examples.append({"question": item["question"], "answer": answer})
    return examples[:limit]


def make_prompts(template: str, examples: list[dict[str, str]]):
    prompts = [template.replace("{question}", item["question"]) for item in examples]
    answers = [item["answer"] for item in examples]
    return prompts, answers


def generate(server, prompts: list[str], seed: int, n: int = 1):
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params={
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "n": n,
            "seed": seed,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        },
        batch_size=64,
    )
    return [completion.text for completion in completions]


def evaluate(
    server,
    tokenizer,
    template: str,
    examples: list[dict[str, str]],
    step: int,
):
    prompts, answers = make_prompts(template, examples)
    responses = generate(server, prompts, SEED + 10000 + step)
    scores = [
        r1_zero_reward_fn(response, answer)
        for response, answer in zip(responses, answers)
    ]
    response_tokenized = tokenizer(responses, add_special_tokens=False)
    average_response_length = sum(
        len(ids) for ids in response_tokenized["input_ids"]
    ) / len(responses)
    return {
        "val_reward": sum(score["reward"] for score in scores) / len(scores),
        "val_format_reward": sum(score["format_reward"] for score in scores) / len(scores),
        "val_answer_reward": sum(score["answer_reward"] for score in scores) / len(scores),
        "val_avg_response_length": average_response_length,
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    with open(PROMPT_PATH, "r", encoding="utf-8") as file:
        prompt_template = file.read()

    train_data = load_examples(TRAIN_PATH, N_TRAIN_EXAMPLES)
    val_data = load_examples(VAL_PATH, N_VAL_EXAMPLES)
    random.shuffle(train_data)

    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(POLICY_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy.config.use_cache = False

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    server = VLLMServer(
        model_id=MODEL_ID,
        host="localhost",
        port=VLLM_PORT,
        gpu=VLLM_GPU,
        seed=SEED,
        logging_level="INFO",
        gpu_memory_utilization=0.8,
    )
    server.start()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / f"config_seed{SEED}.json").open(
        "w", encoding="utf-8"
    ) as config_file:
        json.dump(
            {
                "variant": VARIANT,
                "seed": SEED,
                "model_id": MODEL_ID,
                "prompt_path": PROMPT_PATH,
                "n_train_examples": N_TRAIN_EXAMPLES,
                "n_val_examples": N_VAL_EXAMPLES,
                "num_rollout_steps": NUM_ROLLOUT_STEPS,
                "rollout_batch_size": ROLLOUT_BATCH_SIZE,
                "group_size": GROUP_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "max_grad_norm": MAX_GRAD_NORM,
                "eval_every": EVAL_EVERY,
                "variant_config": VARIANT_CONFIG,
            },
            config_file,
            indent=2,
        )
    metrics_file = (OUTPUT_DIR / f"metrics_seed{SEED}.jsonl").open(
        "w", encoding="utf-8"
    )
    rollouts_file = (OUTPUT_DIR / f"rollouts_seed{SEED}.jsonl").open(
        "w", encoding="utf-8"
    )

    try:
        server.init_weight_sync(POLICY_DEVICE)
        prompts_per_batch = ROLLOUT_BATCH_SIZE // GROUP_SIZE

        for step in range(NUM_ROLLOUT_STEPS):
            batch = random.sample(train_data, prompts_per_batch)
            prompts, answers = make_prompts(prompt_template, batch)
            repeated_prompts = [p for p in prompts for _ in range(GROUP_SIZE)]
            repeated_answers = [a for a in answers for _ in range(GROUP_SIZE)]

            # Generate with the current policy.
            server.sync_policy_weights(policy)
            # Ask vLLM for GROUP_SIZE independent samples per prompt.
            responses = generate(server, prompts, SEED + step, n=GROUP_SIZE)

            if (step + 1) % 40 == 0:
                for prompt, response, answer in zip(
                    repeated_prompts, responses, repeated_answers
                ):
                    rollouts_file.write(
                        json.dumps(
                            {
                                "step": step + 1,
                                "prompt": prompt,
                                "response": response,
                                "ground_truth": answer,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                rollouts_file.flush()

            _, train_metrics = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                max_grad_norm=MAX_GRAD_NORM,
                reward_fn=r1_zero_reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=responses,
                repeated_ground_truths=repeated_answers,
                group_size=GROUP_SIZE,
                **VARIANT_CONFIG,
                importance_reweighting_method="none",
            )

            metrics = {
                "step": step + 1,
                "variant": VARIANT,
                "seed": SEED,
                **train_metrics,
            }
            if (step + 1) % EVAL_EVERY == 0:
                server.sync_policy_weights(policy)
                metrics.update(
                    evaluate(
                        server,
                        tokenizer,
                        prompt_template,
                        val_data,
                        step + 1,
                    )
                )
                print(metrics)
            else:
                print(
                    f"step={step + 1} "
                    f"train_reward={train_metrics['mean_reward']:.4f} "
                    f"loss={train_metrics['loss']:.4f}"
                )

            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
    finally:
        metrics_file.close()
        rollouts_file.close()
        server.stop()


if __name__ == "__main__":
    main()
